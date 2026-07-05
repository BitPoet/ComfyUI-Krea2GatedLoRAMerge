# ComfyUI-Krea2GatedLoRAMerge

Experimental file-level Krea2 LoRA merger with conservative handling for gate layers.

The node writes a new `.safetensors` LoRA into ComfyUI's output folder. Normal layers can be merged by concatenating LoRA ranks, which exactly represents `strength_a * delta_a + strength_b * delta_b`, or compressed back to the larger input rank.

Gate layers can use a conflict rule when the two reconstructed deltas point in opposite directions:

- `pick_strongest`: keep the gate delta with larger norm
- `zero_out`: drop that gate delta

Gate layers are keys containing `.attn.to_gate.` / `.attn.gate.` or `.ff.gate.` / `.mlp.gate.`.

`balance_mode` controls whether matched layers are strength-balanced before merging:

- `off`: use the requested LoRA strengths directly
- `per_layer_norm`: equalize each pair of matched layer delta norms after the requested strengths are applied

`gate_compression` controls whether non-conflicting gate merges remain exact rank concatenations or are compressed back down to `gate_rank`:

- `off`: exact rank concatenation for non-conflicting gates
- `attn_to_gate_only`: compress attention gates only
- `all_gates`: compress attention and MLP gates

Compression is done in low-rank factor space, not by materializing the huge full gate matrix. `compression_device=cuda` can speed this up if enough VRAM is available.

`normal_compression` controls ordinary non-gate layers:

- `off`: exact rank concatenation, larger output
- `max_input_rank`: compress each merged normal layer back to the larger input rank
