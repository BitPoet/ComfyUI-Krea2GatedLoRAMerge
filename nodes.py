import os

import torch
import folder_paths
import comfy.utils
from safetensors.torch import save_file


LORA_A_SUFFIX = ".lora_A.weight"
LORA_B_SUFFIX = ".lora_B.weight"


def _lora_choices():
    return folder_paths.get_filename_list("loras")


def _target_from_a_key(key):
    if key.endswith(LORA_A_SUFFIX):
        return key[:-len(LORA_A_SUFFIX)]
    return None


def _alpha_for(sd, target, rank):
    alpha = sd.get(f"{target}.alpha", None)
    if alpha is None:
        return float(rank)
    return float(alpha.item())


def _is_gate_target(target):
    return (
        ".attn.to_gate" in target or ".attn.gate" in target or
        ".ff.gate" in target or ".mlp.gate" in target
    )


def _lora_factors(sd, target, strength):
    a = sd[f"{target}{LORA_A_SUFFIX}"].float()
    b = sd[f"{target}{LORA_B_SUFFIX}"].float()
    rank = a.shape[0]
    scale = _alpha_for(sd, target, rank) / rank
    return a, b * (float(strength) * scale)


def _factor_dot(a_a, b_a, a_b, b_b):
    return ((b_a.T @ b_b) * (a_a @ a_b.T)).sum()


def _factor_norm(a, b):
    return torch.sqrt(torch.clamp(_factor_dot(a, b, a, b), min=0.0))


def _balanced_strengths(sd_a, sd_b, target, strength_a, strength_b, balance_mode):
    if balance_mode != "per_layer_norm":
        return strength_a, strength_b, False

    a_a, b_a = _lora_factors(sd_a, target, strength_a)
    a_b, b_b = _lora_factors(sd_b, target, strength_b)
    norm_a = _factor_norm(a_a, b_a).item()
    norm_b = _factor_norm(a_b, b_b).item()
    if norm_a == 0.0 or norm_b == 0.0:
        return strength_a, strength_b, False

    target_norm = (norm_a + norm_b) * 0.5
    return float(strength_a) * target_norm / norm_a, float(strength_b) * target_norm / norm_b, True


def _compress_factors(a, b, rank, device_mode):
    current_rank = a.shape[0]
    rank = max(1, min(int(rank), current_rank))
    if rank >= current_rank:
        return a.contiguous(), b.contiguous()

    device = torch.device("cpu")
    if device_mode == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    a_work = a.to(device)
    b_work = b.to(device)

    q, r = torch.linalg.qr(b_work, mode="reduced")
    u, s, vh = torch.linalg.svd(r @ a_work, full_matrices=False)
    u = u[:, :rank]
    s = s[:rank]
    vh = vh[:rank]
    root = torch.sqrt(s)
    b_out = q @ (u * root.unsqueeze(0))
    a_out = root.unsqueeze(1) * vh
    return a_out.cpu().contiguous(), b_out.cpu().contiguous()


def _concat_lora(sd_a, sd_b, target, strength_a, strength_b):
    a_a = sd_a[f"{target}{LORA_A_SUFFIX}"].float()
    b_a = sd_a[f"{target}{LORA_B_SUFFIX}"].float()
    a_b = sd_b[f"{target}{LORA_A_SUFFIX}"].float()
    b_b = sd_b[f"{target}{LORA_B_SUFFIX}"].float()

    scale_a = float(strength_a) * _alpha_for(sd_a, target, a_a.shape[0]) / a_a.shape[0]
    scale_b = float(strength_b) * _alpha_for(sd_b, target, a_b.shape[0]) / a_b.shape[0]
    a = torch.cat([a_a, a_b], dim=0)
    b = torch.cat([b_a * scale_a, b_b * scale_b], dim=1)
    return a.contiguous(), b.contiguous()


def _max_input_rank(sd_a, sd_b, target):
    return max(sd_a[f"{target}{LORA_A_SUFFIX}"].shape[0], sd_b[f"{target}{LORA_A_SUFFIX}"].shape[0])


def _single_lora(sd, target, strength):
    a, b = _lora_factors(sd, target, strength)
    return a.clone().contiguous(), b.clone().contiguous()


def _zero_lora(sd_a, sd_b, target, rank):
    if rank <= 0:
        rank = max(sd_a[f"{target}{LORA_A_SUFFIX}"].shape[0], sd_b[f"{target}{LORA_A_SUFFIX}"].shape[0])
    in_dim = sd_a[f"{target}{LORA_A_SUFFIX}"].shape[1]
    out_dim = sd_a[f"{target}{LORA_B_SUFFIX}"].shape[0]
    a = torch.zeros((rank, in_dim), dtype=torch.float32)
    b = torch.zeros((out_dim, rank), dtype=torch.float32)
    return a, b


def _should_compress_gate(target, compression_mode):
    if compression_mode == "all_gates":
        return True
    if compression_mode == "attn_to_gate_only":
        return ".attn.to_gate" in target or ".attn.gate" in target
    return False


def _gate_lora(sd_a, sd_b, target, strength_a, strength_b, opposite_strategy, opposite_threshold, gate_rank, compression_mode, compression_device):
    a_a, b_a = _lora_factors(sd_a, target, strength_a)
    a_b, b_b = _lora_factors(sd_b, target, strength_b)
    norm_a = _factor_norm(a_a, b_a)
    norm_b = _factor_norm(a_b, b_b)
    if norm_a.item() == 0.0:
        a, b = _single_lora(sd_b, target, strength_b)
        conflict = False
    elif norm_b.item() == 0.0:
        a, b = _single_lora(sd_a, target, strength_a)
        conflict = False
    else:
        cosine = _factor_dot(a_a, b_a, a_b, b_b) / (norm_a * norm_b)
        conflict = bool(cosine.item() < -float(opposite_threshold))
        if conflict:
            if opposite_strategy == "zero_out":
                a, b = _zero_lora(sd_a, sd_b, target, gate_rank)
            else:
                a, b = _single_lora(sd_a, target, strength_a) if norm_a >= norm_b else _single_lora(sd_b, target, strength_b)
        else:
            a, b = _concat_lora(sd_a, sd_b, target, strength_a, strength_b)
            if _should_compress_gate(target, compression_mode):
                a, b = _compress_factors(a, b, gate_rank, compression_device)
    return a, b, conflict


def _copy_unpaired(out, sd, target, strength):
    a, b = _single_lora(sd, target, strength)
    out[f"{target}{LORA_A_SUFFIX}"] = a
    out[f"{target}{LORA_B_SUFFIX}"] = b


def merge_krea2_loras(sd_a, sd_b, strength_a, strength_b, gate_strength, opposite_strategy, opposite_threshold, gate_rank, gate_compression="off", compression_device="cpu", normal_compression="off", balance_mode="off"):
    targets_a = {_target_from_a_key(k) for k in sd_a if _target_from_a_key(k) is not None}
    targets_b = {_target_from_a_key(k) for k in sd_b if _target_from_a_key(k) is not None}
    out = {}
    stats = {"normal": 0, "gate": 0, "gate_conflicts": 0, "balanced": 0, "only_a": 0, "only_b": 0}

    for target in sorted(targets_a | targets_b):
        has_a = target in targets_a and f"{target}{LORA_B_SUFFIX}" in sd_a
        has_b = target in targets_b and f"{target}{LORA_B_SUFFIX}" in sd_b
        if has_a and has_b:
            target_strength_a, target_strength_b, balanced = _balanced_strengths(sd_a, sd_b, target, strength_a, strength_b, balance_mode)
            stats["balanced"] += int(balanced)
            if _is_gate_target(target):
                a, b, conflict = _gate_lora(
                    sd_a, sd_b, target,
                    float(target_strength_a) * float(gate_strength),
                    float(target_strength_b) * float(gate_strength),
                    opposite_strategy,
                    opposite_threshold,
                    gate_rank,
                    gate_compression,
                    compression_device,
                )
                stats["gate"] += 1
                stats["gate_conflicts"] += int(conflict)
            else:
                a, b = _concat_lora(sd_a, sd_b, target, target_strength_a, target_strength_b)
                if normal_compression == "max_input_rank":
                    a, b = _compress_factors(a, b, _max_input_rank(sd_a, sd_b, target), compression_device)
                stats["normal"] += 1
            out[f"{target}{LORA_A_SUFFIX}"] = a
            out[f"{target}{LORA_B_SUFFIX}"] = b
        elif has_a:
            _copy_unpaired(out, sd_a, target, strength_a)
            stats["only_a"] += 1
        elif has_b:
            _copy_unpaired(out, sd_b, target, strength_b)
            stats["only_b"] += 1

    return out, stats


class Krea2GateAwareLoRAMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_a": (_lora_choices(),),
                "lora_b": (_lora_choices(),),
                "strength_a": ("FLOAT", {"default": 0.5, "min": -4.0, "max": 4.0, "step": 0.01}),
                "strength_b": ("FLOAT", {"default": 0.5, "min": -4.0, "max": 4.0, "step": 0.01}),
                "gate_strength": ("FLOAT", {"default": 0.5, "min": -4.0, "max": 4.0, "step": 0.01}),
                "opposite_gate_deltas": (["pick_strongest", "zero_out"], {"default": "pick_strongest"}),
                "opposite_threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "gate_rank": ("INT", {"default": 32, "min": 0, "max": 256}),
                "balance_mode": (["off", "per_layer_norm"], {"default": "off"}),
                "gate_compression": (["off", "attn_to_gate_only", "all_gates"], {"default": "attn_to_gate_only"}),
                "normal_compression": (["off", "max_input_rank"], {"default": "off"}),
                "compression_device": (["cpu", "cuda"], {"default": "cpu"}),
                "filename_prefix": ("STRING", {"default": "loras/krea2_merged"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "merge"
    OUTPUT_NODE = True
    CATEGORY = "model/merging/krea2"

    def merge(self, lora_a, lora_b, strength_a, strength_b, gate_strength, opposite_gate_deltas, opposite_threshold, gate_rank, balance_mode="off", gate_compression="attn_to_gate_only", normal_compression="off", compression_device="cpu", filename_prefix="loras/krea2_merged"):
        path_a = folder_paths.get_full_path_or_raise("loras", lora_a)
        path_b = folder_paths.get_full_path_or_raise("loras", lora_b)
        sd_a = comfy.utils.load_torch_file(path_a, safe_load=True)
        sd_b = comfy.utils.load_torch_file(path_b, safe_load=True)

        merged, stats = merge_krea2_loras(
            sd_a, sd_b, strength_a, strength_b, gate_strength,
            opposite_gate_deltas, opposite_threshold, gate_rank, gate_compression, compression_device, normal_compression, balance_mode,
        )
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
        output_checkpoint = f"{filename}_{counter:05}_.safetensors"
        output_path = os.path.join(full_output_folder, output_checkpoint)
        save_file(merged, output_path, metadata={
            "merge_node": "Krea2GateAwareLoRAMerge",
            "lora_a": lora_a,
            "lora_b": lora_b,
            "strength_a": str(strength_a),
            "strength_b": str(strength_b),
            "gate_strength": str(gate_strength),
            "opposite_gate_deltas": opposite_gate_deltas,
            "balance_mode": balance_mode,
            "gate_compression": gate_compression,
            "normal_compression": normal_compression,
            "compression_device": compression_device,
        })

        rel_path = os.path.join(subfolder, output_checkpoint) if subfolder else output_checkpoint
        info = (
            f"Saved {rel_path}. "
            f"normal={stats['normal']}, gate={stats['gate']}, "
            f"gate_conflicts={stats['gate_conflicts']}, balanced={stats['balanced']}, "
            f"only_a={stats['only_a']}, only_b={stats['only_b']}."
        )
        return {"ui": {"text": [info]}, "result": (info,)}


NODE_CLASS_MAPPINGS = {
    "Krea2GateAwareLoRAMerge": Krea2GateAwareLoRAMerge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2GateAwareLoRAMerge": "Krea2 Gate-Aware LoRA Merge",
}
