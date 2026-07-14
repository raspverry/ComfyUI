# LTX-2.3 MLX Creative Stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Apple Silicon ComfyUI stack that generates photoreal adult-woman stills with Z-Image-Turbo or FLUX.2 Klein and turns them into synchronized LTX-2.3 video through native MLX nodes.

**Architecture:** Keep ComfyUI core at upstream `master`. Extend the existing `dgrauet/ComfyUI-LTXVideo-mlx` fork and the adjacent `dgrauet/ltx-2-mlx` checkout with the smallest callback and Ingredients changes required for progress, cancellation, MPS-to-MLX memory handoff, and current LTX-2.3 reference-sheet inference. Store two complete workflow JSON files in ComfyUI and install only the four model families used by those graphs.

**Tech Stack:** Python 3.12, ComfyUI 0.27.x, PyTorch MPS/BF16, MLX, `ltx-2-mlx` 0.14.18, pytest, ComfyUI workflow JSON.

---

### Task 1: Add optional sampler step callbacks to `ltx-2-mlx`

**Files:**
- Modify: `/Users/hansol/.config/superpowers/worktrees/ltx-2-mlx/comfy-progress/packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/utils/samplers.py`
- Test: `/Users/hansol/.config/superpowers/worktrees/ltx-2-mlx/comfy-progress/tests/test_sampler_step_callback.py`

- [ ] **Step 1: Write the failing callback iterator tests**

```python
from ltx_pipelines_mlx.utils.samplers import _iter_steps


def test_iter_steps_reports_each_step_and_completion():
    calls = []
    values = list(_iter_steps(["a", "b"], 2, lambda current, total: calls.append((current, total))))

    assert values == ["a", "b"]
    assert calls == [(0, 2), (1, 2), (2, 2)]


def test_iter_steps_does_not_require_callback():
    assert list(_iter_steps([1, 2], 2, None)) == [1, 2]
```

- [ ] **Step 2: Run the test and verify the missing helper failure**

Run: `uv run pytest tests/test_sampler_step_callback.py -q`

Expected: collection fails because `_iter_steps` is not defined.

- [ ] **Step 3: Add the minimal callback iterator and optional arguments**

```python
from collections.abc import Callable, Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")
StepCallback = Callable[[int, int], None]


def _iter_steps(values: Iterable[T], total: int, callback: StepCallback | None) -> Iterator[T]:
    for index, value in enumerate(values):
        if callback is not None:
            callback(index, total)
        yield value
    if callback is not None:
        callback(total, total)
```

Add `step_callback: StepCallback | None = None` to `denoise_loop`, `res2s_denoise_loop`, and `guided_denoise_loop`, and wrap each existing tqdm iterator with `_iter_steps(iterator, total_steps, step_callback)`. Preserve every existing default and return type.

- [ ] **Step 4: Run focused and upstream tests**

Run: `uv run pytest tests/test_sampler_step_callback.py tests/test_two_stage.py tests/test_iclora_dev_lora.py -q`

Expected: all selected tests pass.

- [ ] **Step 5: Commit the callback API**

```bash
git add packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/utils/samplers.py tests/test_sampler_step_callback.py
git commit -m "Add sampler progress callbacks"
```

### Task 2: Integrate Comfy memory handoff, progress, and cancellation

**Files:**
- Modify: `custom_nodes/ComfyUI-LTXVideo-mlx/requirements.txt`
- Modify: `custom_nodes/ComfyUI-LTXVideo-mlx/mlx_nodes/mlx_utils.py`
- Modify: `custom_nodes/ComfyUI-LTXVideo-mlx/mlx_nodes/mlx_sampler.py`
- Test: `custom_nodes/ComfyUI-LTXVideo-mlx/tests/test_mlx_runtime.py`

- [ ] **Step 1: Write failing runtime adapter tests**

```python
from mlx_nodes.mlx_utils import ComfyMLXProgress, release_comfy_models


def test_release_comfy_models_unloads_and_empties_cache(monkeypatch):
    calls = []
    monkeypatch.setattr("comfy.model_management.unload_all_models", lambda: calls.append("unload"))
    monkeypatch.setattr("comfy.model_management.soft_empty_cache", lambda force=False: calls.append(("empty", force)))

    release_comfy_models()

    assert calls == ["unload", ("empty", True)]


def test_progress_checks_interrupt_and_updates(monkeypatch):
    calls = []
    monkeypatch.setattr("comfy.model_management.throw_exception_if_processing_interrupted", lambda: calls.append("check"))
    progress = ComfyMLXProgress(total=4, progress_factory=lambda total: type("P", (), {"update_absolute": lambda self, value: calls.append(value)})())

    progress(2, 4)

    assert calls == ["check", 2]
```

- [ ] **Step 2: Verify the tests fail because the adapters are missing**

Run: `python -m pytest tests/test_mlx_runtime.py -q`

Expected: import fails for `ComfyMLXProgress` and `release_comfy_models`.

- [ ] **Step 3: Implement the adapters and wire the samplers**

```python
class ComfyMLXProgress:
    def __init__(self, total, progress_factory=None):
        from comfy.utils import ProgressBar

        self.total = total
        self.progress = (progress_factory or ProgressBar)(total)

    def __call__(self, current, total):
        from comfy.model_management import throw_exception_if_processing_interrupted

        throw_exception_if_processing_interrupted()
        self.progress.update_absolute(min(current, self.total))


def release_comfy_models():
    from comfy import model_management

    model_management.unload_all_models()
    model_management.soft_empty_cache(force=True)
```

Call `release_comfy_models()` before loading the first MLX transformer in each sampler. Create one `ComfyMLXProgress` per sampler execution and pass it as `step_callback` to MLX denoise functions. Pin both MLX packages in `requirements.txt` to `v0.14.18`.

- [ ] **Step 4: Run focused tests and import the custom node in ComfyUI**

Run: `python -m pytest tests/test_mlx_runtime.py -q`

Run: `python -c "import importlib.util, pathlib; p=pathlib.Path('__init__.py'); s=importlib.util.spec_from_file_location('comfy_ltx_mlx', p, submodule_search_locations=[str(p.parent)]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(len(m.NODE_CLASS_MAPPINGS))"`

Expected: tests pass and node mappings print without an import error.

- [ ] **Step 5: Commit the runtime integration**

```bash
git add requirements.txt mlx_nodes/mlx_utils.py mlx_nodes/mlx_sampler.py tests/test_mlx_runtime.py
git commit -m "Connect MLX samplers to Comfy runtime"
```

### Task 3: Add the LTX-2.3 Ingredients single-stage recipe

**Files:**
- Modify: `custom_nodes/ComfyUI-LTXVideo-mlx/mlx_nodes/mlx_loader.py`
- Modify: `custom_nodes/ComfyUI-LTXVideo-mlx/mlx_nodes/mlx_sampler.py`
- Modify: `custom_nodes/ComfyUI-LTXVideo-mlx/mlx_nodes/__init__.py`
- Test: `custom_nodes/ComfyUI-LTXVideo-mlx/tests/test_ingredients_config.py`

- [ ] **Step 1: Write failing recipe and validation tests**

```python
import pytest

from mlx_nodes.mlx_sampler import IngredientsConfig, validate_video_shape


def test_ingredients_defaults_match_current_single_stage_recipe():
    config = IngredientsConfig()
    assert config.width == 768
    assert config.height == 448
    assert config.num_frames == 121
    assert config.frame_rate == 24
    assert config.steps == 8
    assert config.distilled_lora_strength == 0.5


def test_video_shape_requires_8k_plus_1_frames():
    with pytest.raises(ValueError, match="8k \\+ 1"):
        validate_video_shape(768, 448, 120, single_stage=True)


def test_single_stage_requires_dimensions_divisible_by_32():
    with pytest.raises(ValueError, match="divisible by 32"):
        validate_video_shape(769, 448, 121, single_stage=True)
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_ingredients_config.py -q`

Expected: import fails because `IngredientsConfig` and `validate_video_shape` do not exist.

- [ ] **Step 3: Implement the recipe with no generic pipeline abstraction**

```python
@dataclass(frozen=True)
class IngredientsConfig:
    width: int = 768
    height: int = 448
    num_frames: int = 121
    frame_rate: int = 24
    steps: int = 8
    dev_transformer: str = "transformer-dev.safetensors"
    distilled_lora: str = "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
    distilled_lora_strength: float = 0.5


def validate_video_shape(width, height, num_frames, *, single_stage):
    divisor = 32 if single_stage else 64
    if width % divisor or height % divisor:
        raise ValueError(f"width and height must be divisible by {divisor}")
    if (num_frames - 1) % 8:
        raise ValueError("num_frames must follow 8k + 1")
```

Add `LTXVMLXIngredientsSampler` as a dedicated node. It must load `transformer-dev.safetensors`, fuse the distilled LoRA at `0.5` and the Ingredients IC-LoRA at the workflow-provided strength, run the fixed distilled sigma schedule once at full resolution, strip appended reference tokens, and return Comfy `IMAGE` plus `AUDIO`. Reuse the existing VAE/reference encode and LoRA fusion code; do not add a subprocess path.

- [ ] **Step 4: Run Ingredients and runtime tests**

Run: `python -m pytest tests/test_ingredients_config.py tests/test_mlx_runtime.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit Ingredients support**

```bash
git add mlx_nodes/mlx_loader.py mlx_nodes/mlx_sampler.py mlx_nodes/__init__.py tests/test_ingredients_config.py
git commit -m "Add LTX 2.3 Ingredients sampler"
```

### Task 4: Build and validate the two complete workflows

**Files:**
- Create: `scripts/ltx_stack/build_workflows.py`
- Create: `workflows/ltx-mlx/quick-i2v.json`
- Create: `workflows/ltx-mlx/recurring-character.json`
- Create: `tests-unit/ltx_stack/test_workflows.py`

- [ ] **Step 1: Write failing structural workflow tests**

```python
import json
from pathlib import Path

WORKFLOWS = Path("workflows/ltx-mlx")


def load(name):
    return json.loads((WORKFLOWS / name).read_text())


def node_types(workflow):
    return {node["type"] for node in workflow["nodes"]}


def test_quick_i2v_contains_z_image_and_mlx_two_stage():
    types = node_types(load("quick-i2v.json"))
    assert "LTXVMLXTwoStageSampler" in types
    assert "SaveImage" in types
    assert "SaveVideo" in types


def test_recurring_character_contains_multi_reference_and_ingredients():
    workflow = load("recurring-character.json")
    types = node_types(workflow)
    assert "RepeatImageBatch" in types
    assert "LTXVMLXIngredientsSampler" in types
    assert any(subgraph["name"].startswith("Image Edit (Flux.2 Klein") for subgraph in workflow["definitions"]["subgraphs"])
```

- [ ] **Step 2: Verify the missing-workflow failure**

Run: `python -m pytest tests-unit/ltx_stack/test_workflows.py -q`

Expected: both tests fail with `FileNotFoundError`.

- [ ] **Step 3: Implement the deterministic workflow builder**

The builder must copy the installed official `image_z_image_turbo.json` and `image_flux2_klein_image_edit_4b_distilled.json` definitions, preserve their subgraph IDs, add only the MLX loader/encoder/sampler/video-save nodes, connect the still or reference-sheet output to MLX, set `dgrauet/ltx-2.3-mlx-q8`, and write version `0.4` workflow JSON with stable IDs. Defaults must explicitly describe a `30-year-old adult woman`; the video prompt must focus on motion, camera, dialogue, and ambient audio.

- [ ] **Step 4: Generate and validate both workflows**

Run: `python scripts/ltx_stack/build_workflows.py`

Run: `python -m pytest tests-unit/ltx_stack/test_workflows.py -q`

Expected: both workflow tests pass.

- [ ] **Step 5: Commit workflow artifacts**

```bash
git add scripts/ltx_stack/build_workflows.py workflows/ltx-mlx tests-unit/ltx_stack/test_workflows.py
git commit -m "Add MLX image to video workflows"
```

### Task 5: Provision models and final runtime entrypoint

**Files:**
- Create: `scripts/ltx_stack/model_manifest.json`
- Create: `scripts/ltx_stack/verify_install.py`
- Create: `scripts/start_ltx_stack_macos.sh`
- Create: `docs/ltx-mlx-stack.md`
- Test: `tests-unit/ltx_stack/test_model_manifest.py`

- [ ] **Step 1: Write the failing manifest test**

```python
import json
from pathlib import Path


def test_model_manifest_uses_only_selected_model_families():
    manifest = json.loads(Path("scripts/ltx_stack/model_manifest.json").read_text())
    assert set(manifest) == {"z_image_turbo", "flux2_klein_4b", "ltx_2_3_q8", "ingredients"}
    assert manifest["ltx_2_3_q8"]["repo_id"] == "dgrauet/ltx-2.3-mlx-q8"
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests-unit/ltx_stack/test_model_manifest.py -q`

Expected: `model_manifest.json` is missing.

- [ ] **Step 3: Add the exact manifest, startup script, and concise operator guide**

The manifest must list exact Hugging Face repo IDs, filenames, destination directories, and license URLs. The startup script must execute the repository `.venv/bin/python main.py --listen 127.0.0.1 --port 8188 --preview-method auto` and fail clearly if the environment or custom node is missing. The guide must document the two workflows, q8 standard-memory choice, the gated Ingredients download, and the LTX `$10M` commercial-license threshold.

- [ ] **Step 4: Download and place the selected model files**

Use `huggingface_hub.hf_hub_download` for individual Comfy model files and `snapshot_download` for the q8 MLX repository. Reuse the existing Hugging Face cache and symlink or copy only the Comfy split files into `models/`.

- [ ] **Step 5: Validate manifests and model presence**

Run: `python -m pytest tests-unit/ltx_stack/test_model_manifest.py -q`

Run: `python scripts/ltx_stack/verify_install.py`

Expected: the manifest test passes and every required local artifact is reported present.

- [ ] **Step 6: Commit runtime setup files**

```bash
git add scripts/ltx_stack/model_manifest.json scripts/start_ltx_stack_macos.sh docs/ltx-mlx-stack.md tests-unit/ltx_stack/test_model_manifest.py
git commit -m "Add macOS LTX stack setup"
```

### Task 6: End-to-end verification and integration

**Files:**
- Modify only files required by failures found during verification.

- [ ] **Step 1: Run all local stack tests**

Run: `python -m pytest tests-unit/ltx_stack custom_nodes/ComfyUI-LTXVideo-mlx/tests -q`

Expected: all tests pass.

- [ ] **Step 2: Run the upstream ComfyUI unit suite**

Run: `python -m pytest tests-unit -q`

Expected: at least the baseline `1098 passed, 10 skipped` with no failures.

- [ ] **Step 3: Start ComfyUI and verify node registration over the local API**

Run: `./scripts/start_ltx_stack_macos.sh`

Run: `curl -fsS http://127.0.0.1:8188/object_info/LTXVMLXIngredientsSampler`

Expected: HTTP 200 and the node schema contains the Ingredients defaults.

- [ ] **Step 4: Use the in-app browser to import both workflow JSON files**

Verify that no node is missing, model widgets resolve to installed filenames, links are intact, and queueing the smallest allowed smoke configurations produces one PNG and one video artifact.

- [ ] **Step 5: Record actual benchmark evidence**

Record model, quantization, width, height, frames, steps, wall time, and peak process memory in `docs/ltx-mlx-stack.md`. Do not claim Ingredients identity quality until its gated checkpoint completes a real 121-frame run.

- [ ] **Step 6: Merge the reviewed branches into the fresh root**

Merge `codex/ltx-mlx-stack` into the root ComfyUI checkout and the callback branch into `/Users/hansol/dev/oss/ltx-2-mlx`. Copy the tested custom-node checkout and user workflow files into the final root, then rerun the install verifier from `/Users/hansol/dev/oss/comfy`.
