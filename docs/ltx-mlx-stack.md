# LTX 2.3 MLX stack on macOS

This stack keeps image generation in ComfyUI and runs LTX 2.3 video generation in-process with MLX. It standardizes on `dgrauet/ltx-2.3-mlx-q8`: q8 is the quality/memory profile used by both workflows, while q4 and other variants are intentionally outside this setup.

## Workflows

- `workflows/ltx-mlx/quick-i2v.json` creates a still with Z-Image-Turbo and sends it to the LTX MLX image-to-video sampler.
- `workflows/ltx-mlx/recurring-character.json` uses FLUX.2 Klein multi-reference editing to build a reference sheet, repeats it across the video timeline, and runs the LTX Ingredients sampler.

Import either JSON file from the ComfyUI workflow menu. The recurring-character workflow remains unavailable until the gated Ingredients checkpoint is installed.

## Install check and startup

The exact repositories, filenames, and destinations are recorded in `scripts/ltx_stack/model_manifest.json`. Place or symlink the curated q8 files under `models/ltx/ltx-2.3-mlx-q8`; using this local directory avoids downloading unused older transformers and upscalers from the full repository snapshot. Keep the required `mlx-community/gemma-3-12b-it-4bit` text encoder in the normal Hugging Face cache. The Qwen 3 4B text encoder is shared by Z-Image-Turbo and FLUX.2 Klein and should exist only once at `models/text_encoders/qwen_3_4b.safetensors`.

Run the offline verifier before starting ComfyUI:

```bash
.venv/bin/python scripts/ltx_stack/verify_install.py
./scripts/start_ltx_stack_macos.sh
```

The verifier fails for a missing public model, incomplete curated q8 directory, incomplete Gemma snapshot, or missing custom node. Missing Ingredients weights are reported as optional because Hugging Face authentication alone is insufficient until the repository terms have been accepted.

For Ingredients, accept access at [LTX-2.3 Ingredients](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients), authenticate with `hf auth login`, download `ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors`, and place it at `models/loras/ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors`.

## Licenses

Z-Image-Turbo and FLUX.2 Klein 4B identify as Apache 2.0 in their official model cards: [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) and [FLUX.2 Klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B). The MLX Gemma text encoder is subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms). The q8 conversion and Ingredients adapter derive from LTX 2.3 and remain subject to the [LTX-2 Community License Agreement](https://github.com/Lightricks/LTX-2/blob/main/LICENSE).

That agreement states that entities with annual revenue of at least USD 10,000,000 must obtain a paid commercial-use license. It also aggregates entities under common control when applying thresholds. See the license text and [Lightricks commercial licensing contact](https://ltx.io/model/licensing) for the controlling terms; this note is not legal advice.
