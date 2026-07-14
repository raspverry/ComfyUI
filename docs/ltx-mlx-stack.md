# LTX 2.3 MLX stack on macOS

This stack keeps image generation in ComfyUI and runs LTX 2.3 video generation in-process with MLX. It standardizes on `dgrauet/ltx-2.3-mlx-q8`: q8 is the quality/memory profile used by both workflows, while q4 and other variants are intentionally outside this setup. The workflows use the `standard` MLX profile and target Apple Silicon with at least 64 GB of unified memory. On 16 GB or 32 GB systems, change both MLX loader nodes to `low_vram` before queueing. The starter prompt padded sequence is 256 tokens.

## Workflows

- `workflows/ltx-mlx/quick-i2v.json` creates a still with Z-Image-Turbo and sends it to the LTX MLX image-to-video sampler.
- `workflows/ltx-mlx/recurring-character.json` uses FLUX.2 Klein multi-reference editing to build a reference sheet, repeats it across the video timeline, and runs the LTX Ingredients sampler.

Import either JSON file from the ComfyUI workflow menu. The recurring-character workflow remains unavailable until the gated Ingredients checkpoint is installed.

Before queueing the recurring-character workflow, upload two identity references and select them in the LoadImage nodes named `reference-front.png` and `reference-profile.png`. These are placeholders, so the workflow cannot run while either image is missing.

The maintained custom node is [raspverry/ComfyUI-LTXVideo-mlx](https://github.com/raspverry/ComfyUI-LTXVideo-mlx), pinned here to commit `f0e6f3b05661e8a7e515e6f11bd74c8ed4fb688b`. Stock `dgrauet/ComfyUI-LTXVideo-mlx` does not contain the Ingredients node or the in-process I2V fix used by these workflows.

```bash
git clone https://github.com/raspverry/ComfyUI-LTXVideo-mlx.git custom_nodes/ComfyUI-LTXVideo-mlx
git -C custom_nodes/ComfyUI-LTXVideo-mlx checkout f0e6f3b05661e8a7e515e6f11bd74c8ed4fb688b
```

## Install check and startup

The exact repositories, filenames, and destinations are recorded in `scripts/ltx_stack/model_manifest.json`. Place or symlink the curated q8 files under `models/ltx/ltx-2.3-mlx-q8`; using this local directory avoids downloading unused older transformers and upscalers from the full repository snapshot. Keep the required `mlx-community/gemma-3-12b-it-4bit` text encoder in the normal Hugging Face cache. The Qwen 3 4B text encoder is shared by Z-Image-Turbo and FLUX.2 Klein and should exist only once at `models/text_encoders/qwen_3_4b.safetensors`.

Run the offline verifier before starting ComfyUI:

```bash
.venv/bin/python scripts/ltx_stack/verify_install.py
./scripts/start_ltx_stack_macos.sh
```

The startup script exports `HF_HUB_OFFLINE=1`, so the verified pinned Gemma snapshot must already exist in the selected Hugging Face cache before launch. The verifier fails for a missing public model, incomplete curated q8 directory, incomplete Gemma snapshot, or missing custom node. Missing Ingredients weights are reported as optional because Hugging Face authentication alone is insufficient until the repository terms have been accepted; the recurring-character workflow still requires them.

For Ingredients, accept access at [LTX-2.3 Ingredients](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients), authenticate with `hf auth login`, download `ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors`, and place it at `models/loras/ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors`.

## Licenses

Z-Image-Turbo and FLUX.2 Klein 4B identify as Apache 2.0 in their official model cards: [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) and [FLUX.2 Klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B). The MLX Gemma text encoder is subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms). The q8 conversion and Ingredients adapter derive from LTX 2.3 and remain subject to the [LTX-2 Community License Agreement](https://github.com/Lightricks/LTX-2/blob/main/LICENSE).

That agreement states that entities with annual revenue of at least USD 10,000,000 must obtain a paid commercial-use license. It also aggregates entities under common control when applying thresholds. See the license text and [Lightricks commercial licensing contact](https://ltx.io/model/licensing) for the controlling terms; this note is not legal advice.
