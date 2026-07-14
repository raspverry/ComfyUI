# LTX MLX Stack Hardening Design

**Status:** Approved for local implementation. Do not push any repository.

## Goal

Make the local LTX 2.3 MLX stack honest, reproducible once its custom node is published, cheaper to store, and safer to run on Apple Silicon. Keep ComfyUI core changes limited to the stack-owned scripts, tests, workflows, and guide.

## Repository boundaries

- ComfyUI owns workflow artifacts, the model manifest, offline startup, install verification, tests, and operator documentation.
- `ComfyUI-LTXVideo-mlx` owns MLX node implementations. Keep its seven local commits unchanged for this pass. Later publish them from a fork and tag the tested revision.
- `ltx-2-mlx` remains pinned to public release `0.14.18`. Preserve the local callback branch, but do not make the stack depend on an unpublished commit.
- Do not vendor either dependency into ComfyUI and do not push any branch.

## ComfyUI changes

### Install verification

Extend `verify_install.py` with checks that fail before startup when:

- the custom node lacks the MLX loader, two-stage sampler, Ingredients sampler, or in-process I2V preprocessing implementation;
- installed `ltx-core-mlx` or `ltx-pipelines-mlx` is not `0.14.18`;
- a required local artifact is missing or has a different byte size from the manifest;
- the Gemma snapshot is not the pinned snapshot revision or is incomplete.

Keep Ingredients optional until its gated checkpoint is installed. Avoid importing the custom node in the verifier; imports initialize ComfyUI and obscure installation errors. Check narrow source capabilities and installed package metadata instead.

### Offline runtime

Set `HF_HUB_OFFLINE=1` in the stack startup script. The verifier already requires the Gemma snapshot, so runtime downloads and Hub metadata checks are unnecessary. Do not add download behavior.

### Model set

Remove `transformer-distilled-1.1.safetensors` from the curated manifest. Neither shipped workflow uses it; both use `transformer-dev.safetensors` with the distilled LoRA.

Record expected byte sizes for required files. This catches empty and truncated files without hashing roughly 55 GB on every verification run.

### Workflows

- Use `max_length=256` for both starter prompts. Their current prompts fit comfortably and the smaller padded sequence reduces Gemma memory and watchdog risk.
- Leave the standard memory profile because this machine has 128 GB unified memory. Document that the starter workflows target 64 GB or more and that lower-memory systems must select `low_vram`.
- Use an empty negative prompt for the Ingredients workflow. Its sampler ignores negative embeddings, so encoding them only repeats the expensive Gemma pass.
- Keep the two reference image placeholders. Document that users must upload and select both images before queueing; do not add arbitrary identity assets to the repository.

### Tests

Add regression tests for:

- custom-node capability and package-version failures;
- zero-byte or wrong-size required files;
- the pinned Gemma snapshot;
- removal of the unused transformer;
- offline startup;
- workflow regeneration matching checked-in JSON;
- internal and outer subgraph link integrity;
- safe prompt length and empty Ingredients negative prompt.

Run custom-node tests from its repository root rather than using the broken combined-root command.

## Local storage cleanup

After tests no longer require the old artifact:

1. Remove the curated `transformer-distilled-1.1.safetensors` symlink.
2. Remove its snapshot symlink and unique Hugging Face cache blob after verifying no remaining required path resolves to that blob. Expected recovery: about 20.6 GB decimal.
3. Remove the callback worktree's disposable `.venv` after its tests pass. Preserve the worktree, branch, and commit. Expected recovery: about 345 MB.
4. Preserve all required model blobs, the main ComfyUI `.venv`, generated smoke evidence, and the gated Ingredients slot.

## Future fork publication

When publication is authorized:

1. Fork `dgrauet/ComfyUI-LTXVideo-mlx`.
2. Push the reviewed local commits to a named branch.
3. Tag the exact tested revision.
4. Replace the guide's local-only warning with clone and checkout commands pinned to that tag or commit.
5. Update the verifier to require that published revision.

A separate `ltx-2-mlx` fork is unnecessary while the compatibility adapter supports `0.14.18`. Reconsider only when removing that adapter.

## Completion criteria

- Main ComfyUI unit suite and stack tests pass.
- Custom-node and callback-branch focused tests pass from their owning repositories.
- Workflow generation is deterministic and all links validate.
- Offline verifier rejects stale custom nodes, wrong package versions, and invalid model files.
- ComfyUI starts offline and registers every top-level workflow node.
- No tracked changes appear outside the approved stack files and design/plan documents.
- Cleanup reports actual reclaimed disk space.
