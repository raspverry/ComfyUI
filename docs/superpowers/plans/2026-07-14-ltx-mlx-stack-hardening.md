# LTX MLX Stack Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the local LTX 2.3 MLX starter stack fail honestly, run without Hub access, avoid wasted Gemma work, and reclaim storage used by an unreferenced transformer.

**Architecture:** Keep all ComfyUI changes inside the existing LTX stack scripts, workflows, tests, and guide. Verify the ignored custom-node checkout through narrow source capabilities without importing it, keep public `ltx-2-mlx==0.14.18`, and defer fork publication. Use manifest byte sizes and a pinned Gemma snapshot for fast local validation.

**Tech Stack:** Python 3.12, pytest, JSON workflow format, Bash, Hugging Face cache, ComfyUI local API.

---

## File map

- `scripts/ltx_stack/model_manifest.json`: exact required artifacts, byte sizes, and Gemma revision.
- `scripts/ltx_stack/verify_install.py`: offline install, capability, package-version, size, and snapshot checks.
- `tests-unit/ltx_stack/test_model_manifest.py`: verifier and manifest regression coverage.
- `scripts/ltx_stack/build_workflows.py`: safe starter prompt defaults.
- `workflows/ltx-mlx/*.json`: deterministic generated workflow artifacts.
- `tests-unit/ltx_stack/test_workflows.py`: generated-artifact and graph-link contracts.
- `scripts/start_ltx_stack_macos.sh`: network-silent runtime entrypoint.
- `docs/ltx-mlx-stack.md`: local-only, input, memory, and offline operating requirements.
- `docs/superpowers/specs/2026-07-14-ltx-mlx-stack-hardening-design.md`: approved design; do not modify during implementation.

### Task 1: Harden the manifest and offline verifier

**Files:**
- Modify: `scripts/ltx_stack/model_manifest.json`
- Modify: `scripts/ltx_stack/verify_install.py`
- Modify: `tests-unit/ltx_stack/test_model_manifest.py`

- [ ] **Step 1: Replace zero-byte fixtures and add failing verifier tests**

In `tests-unit/ltx_stack/test_model_manifest.py`, import the verifier for the package-version unit test:

```python
from scripts.ltx_stack import verify_install
```

Add these helpers and use them from `create_valid_install`:

```python
CUSTOM_NODE_SOURCES = {
    "__init__.py": "",
    "mlx_nodes/__init__.py": '"LTXVMLXIngredientsSampler"\n',
    "mlx_nodes/mlx_loader.py": (
        "class LTXVMLXCheckpointLoader:\n    pass\n\n"
        "class LTXVMLXTextEncoderLoader:\n    pass\n"
    ),
    "mlx_nodes/mlx_sampler.py": "class LTXVMLXTwoStageSampler:\n    pass\n",
    "mlx_nodes/mlx_utils.py": "def prepare_i2v_image():\n    pass\n",
}


def create_sized_file(path, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    path.truncate(size)


def create_custom_node(root):
    custom_node = root / "custom_nodes/ComfyUI-LTXVideo-mlx"
    for relative, source in CUSTOM_NODE_SOURCES.items():
        path = custom_node / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
```

Replace the file creation inside `create_valid_install` with sized sparse files and the pinned snapshot:

```python
def create_valid_install(root, hf_cache):
    manifest = load_manifest()
    for model in manifest.values():
        if model["required"]:
            for file in model["files"]:
                create_sized_file(root / file["destination"], file["size"])
        for destination in model.get("shared_dependencies", []):
            shared = root / destination
            if not shared.exists():
                shared.parent.mkdir(parents=True, exist_ok=True)
                shared.touch()

    create_custom_node(root)

    text_encoder = manifest["ltx_2_3_q8"]["text_encoder"]
    repo_dir = f"models--{text_encoder['repo_id'].replace('/', '--')}"
    snapshot = hf_cache / repo_dir / "snapshots" / text_encoder["revision"]
    for file in text_encoder["files"]:
        create_sized_file(snapshot / file["filename"], file["size"])
```

Add this helper and use it in `test_manifest_records_exact_comfy_model_files` so location assertions remain focused while sizes receive their own verifier coverage:

```python
def model_locations(model):
    keys = ("filename", "destination", "repo_id")
    return [{key: file[key] for key in keys if key in file} for file in model["files"]]
```

Replace each existing `manifest[<name>]["files"] == [...]` location assertion with `model_locations(manifest[<name>]) == [...]`. Update text-encoder filename assertions to read `file["filename"]` where entries were previously strings. Add these tests:

```python
def run_verifier(root, hf_cache):
    return subprocess.run(
        [sys.executable, str(VERIFY_SCRIPT), "--root", str(root)],
        env=clean_cache_env(root, HF_HUB_CACHE=hf_cache),
        capture_output=True,
        text=True,
    )


def test_ltx_manifest_omits_unused_distilled_transformer():
    filenames = [file["filename"] for file in load_manifest()["ltx_2_3_q8"]["files"]]

    assert "transformer-distilled-1.1.safetensors" not in filenames


def test_verifier_rejects_wrong_sized_required_file(tmp_path):
    hf_cache = tmp_path / "hf-cache"
    create_valid_install(tmp_path, hf_cache)
    (tmp_path / "models/ltx/ltx-2.3-mlx-q8/config.json").write_text("bad")

    result = run_verifier(tmp_path, hf_cache)

    assert result.returncode == 1
    assert "models/ltx/ltx-2.3-mlx-q8/config.json" in result.stdout


def test_verifier_requires_custom_node_capabilities(tmp_path):
    hf_cache = tmp_path / "hf-cache"
    create_valid_install(tmp_path, hf_cache)
    (tmp_path / "custom_nodes/ComfyUI-LTXVideo-mlx/mlx_nodes/__init__.py").write_text("")

    result = run_verifier(tmp_path, hf_cache)

    assert result.returncode == 1
    assert "LTXVMLXIngredientsSampler" in result.stdout


def test_verifier_requires_pinned_gemma_snapshot(tmp_path):
    hf_cache = tmp_path / "hf-cache"
    create_valid_install(tmp_path, hf_cache)
    text_encoder = load_manifest()["ltx_2_3_q8"]["text_encoder"]
    snapshots = hf_cache / f"models--{text_encoder['repo_id'].replace('/', '--')}" / "snapshots"
    (snapshots / text_encoder["revision"]).rename(snapshots / "wrong-revision")

    result = run_verifier(tmp_path, hf_cache)

    assert result.returncode == 1
    assert text_encoder["revision"] in result.stdout


def test_verifier_rejects_wrong_ltx_package_version(monkeypatch, capsys):
    monkeypatch.setattr(verify_install, "version", lambda _name: "0.0.0")

    assert verify_install.verify_package_versions() is False
    assert "0.14.18" in capsys.readouterr().out
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests-unit/ltx_stack/test_model_manifest.py::test_ltx_manifest_omits_unused_distilled_transformer \
  tests-unit/ltx_stack/test_model_manifest.py::test_verifier_rejects_wrong_sized_required_file \
  tests-unit/ltx_stack/test_model_manifest.py::test_verifier_requires_custom_node_capabilities \
  tests-unit/ltx_stack/test_model_manifest.py::test_verifier_requires_pinned_gemma_snapshot \
  tests-unit/ltx_stack/test_model_manifest.py::test_verifier_rejects_wrong_ltx_package_version
```

Expected: failures because the manifest still includes the unused transformer, entries lack sizes/revision, and the verifier lacks capability/package checks.

- [ ] **Step 3: Add exact artifact metadata**

Remove the `transformer-distilled-1.1.safetensors` entry. Add these exact `size` values to the remaining `files` entries in `scripts/ltx_stack/model_manifest.json`:

```python
EXPECTED_DESTINATION_SIZES = {
    "models/diffusion_models/z_image_turbo_bf16.safetensors": 12309866400,
    "models/text_encoders/qwen_3_4b.safetensors": 8044982048,
    "models/vae/ae.safetensors": 335304388,
    "models/diffusion_models/flux-2-klein-4b.safetensors": 7751105712,
    "models/vae/flux2-vae.safetensors": 336213556,
    "models/ltx/ltx-2.3-mlx-q8/config.json": 951,
    "models/ltx/ltx-2.3-mlx-q8/embedded_config.json": 7234,
    "models/ltx/ltx-2.3-mlx-q8/quantize_config.json": 100,
    "models/ltx/ltx-2.3-mlx-q8/split_model.json": 626,
    "models/ltx/ltx-2.3-mlx-q8/connector.safetensors": 6344495512,
    "models/ltx/ltx-2.3-mlx-q8/transformer-dev.safetensors": 20597189549,
    "models/ltx/ltx-2.3-mlx-q8/ltx-2.3-22b-distilled-lora-384-1.1.safetensors": 7605507256,
    "models/ltx/ltx-2.3-mlx-q8/spatial_upscaler_x2_v1_1.safetensors": 995745061,
    "models/ltx/ltx-2.3-mlx-q8/spatial_upscaler_x2_v1_1_config.json": 275,
    "models/ltx/ltx-2.3-mlx-q8/vae_encoder.safetensors": 637885319,
    "models/ltx/ltx-2.3-mlx-q8/vae_decoder.safetensors": 814349531,
    "models/ltx/ltx-2.3-mlx-q8/audio_vae.safetensors": 106509048,
    "models/ltx/ltx-2.3-mlx-q8/vocoder.safetensors": 258313851,
}
```

Remove `"transformer-distilled-1.1.safetensors"` from the expected filename list in `test_ltx_q8_uses_only_the_curated_local_runtime_files`.

Set the text encoder revision and replace its string file list with these objects:

```json
"revision": "86cc6a8dedbc456dd0e4af01a9d09f396f77e558",
"files": [
  {"filename": "added_tokens.json", "size": 35},
  {"filename": "chat_template.json", "size": 1615},
  {"filename": "config.json", "size": 1141},
  {"filename": "generation_config.json", "size": 192},
  {"filename": "model-00001-of-00002.safetensors", "size": 5367455313},
  {"filename": "model-00002-of-00002.safetensors", "size": 2661219935},
  {"filename": "model.safetensors.index.json", "size": 108605},
  {"filename": "preprocessor_config.json", "size": 570},
  {"filename": "processor_config.json", "size": 70},
  {"filename": "special_tokens_map.json", "size": 662},
  {"filename": "tokenizer.json", "size": 33384568},
  {"filename": "tokenizer.model", "size": 4689074},
  {"filename": "tokenizer_config.json", "size": 1157007}
]
```

Keep the optional Ingredients entry without a size until the gated artifact is installed and measured.

- [ ] **Step 4: Implement narrow verifier checks**

Update `scripts/ltx_stack/verify_install.py` imports and constants:

```python
from importlib.metadata import PackageNotFoundError, version


PACKAGE_VERSIONS = {
    "ltx-core-mlx": "0.14.18",
    "ltx-pipelines-mlx": "0.14.18",
}
CUSTOM_NODE_CAPABILITIES = {
    "mlx_nodes/__init__.py": ("LTXVMLXIngredientsSampler",),
    "mlx_nodes/mlx_loader.py": (
        "class LTXVMLXCheckpointLoader",
        "class LTXVMLXTextEncoderLoader",
    ),
    "mlx_nodes/mlx_sampler.py": ("class LTXVMLXTwoStageSampler",),
    "mlx_nodes/mlx_utils.py": ("def prepare_i2v_image",),
}
```

Add these helpers:

```python
def valid_file(path, file):
    if not path.is_file():
        return False
    return "size" not in file or path.stat().st_size == file["size"]


def verify_package_versions():
    valid = True
    for package, expected in PACKAGE_VERSIONS.items():
        try:
            actual = version(package)
        except PackageNotFoundError:
            actual = "missing"
        if actual == expected:
            print(f"[OK] {package}: {actual}")
        else:
            print(f"[MISMATCH] {package}: expected {expected}, found {actual}")
            valid = False
    return valid


def verify_custom_node(root):
    custom_node = root / "custom_nodes/ComfyUI-LTXVideo-mlx"
    missing = []
    for relative, capabilities in CUSTOM_NODE_CAPABILITIES.items():
        path = custom_node / relative
        if not path.is_file():
            missing.append(relative)
            continue
        source = path.read_text()
        missing.extend(capability for capability in capabilities if capability not in source)
    if missing:
        print(f"[MISSING] custom_node: {', '.join(missing)}")
        return False
    print(f"[OK] custom_node: {custom_node}")
    return True
```

Replace `find_snapshot` and `verify_snapshot` with pinned, sized checks:

```python
def find_snapshot(model):
    repo_dir = Path(HF_HUB_CACHE) / f"models--{model['repo_id'].replace('/', '--')}"
    snapshot = repo_dir / "snapshots" / model["revision"]
    if snapshot.is_dir() and all(valid_file(snapshot / file["filename"], file) for file in model["files"]):
        return snapshot
    return None


def verify_snapshot(name, model):
    snapshot = find_snapshot(model)
    if snapshot is None:
        print(f"[MISSING] {name}: snapshot {model['revision']} for {model['repo_id']}")
        return False
    print(f"[OK] {name}: {snapshot}")
    return True
```

In `verify`, replace the `__init__.py` check with `verify_custom_node`, call `verify_package_versions`, and use `valid_file` for manifest entries:

```python
failed = not verify_custom_node(root)
failed = not verify_package_versions() or failed

for name, model in manifest.items():
    missing = [
        file["destination"]
        for file in model["files"]
        if not valid_file(root / file["destination"], file)
    ]
```

Keep shared dependency checks and optional Ingredients behavior unchanged.

- [ ] **Step 5: Run the full manifest suite and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest -q tests-unit/ltx_stack/test_model_manifest.py
.venv/bin/python scripts/ltx_stack/verify_install.py
```

Expected: all manifest tests pass; verifier exits 0 with required artifacts, packages, custom-node capabilities, and pinned Gemma reported OK; Ingredients remains optional.

- [ ] **Step 6: Commit the verifier change**

```bash
git add scripts/ltx_stack/model_manifest.json scripts/ltx_stack/verify_install.py tests-unit/ltx_stack/test_model_manifest.py
git commit -m "Harden LTX MLX install verification"
```

### Task 2: Make workflow defaults safe and test generated graphs

**Files:**
- Modify: `scripts/ltx_stack/build_workflows.py`
- Modify: `tests-unit/ltx_stack/test_workflows.py`
- Regenerate: `workflows/ltx-mlx/quick-i2v.json`
- Regenerate: `workflows/ltx-mlx/recurring-character.json`

- [ ] **Step 1: Add failing default and deterministic-build tests**

Add imports to `tests-unit/ltx_stack/test_workflows.py`:

```python
from importlib.resources import files

from scripts.ltx_stack.build_workflows import build_workflows
```

Add a link normalizer and validate every stored graph:

```python
def link_values(link):
    if isinstance(link, dict):
        return (
            link["id"],
            link["origin_id"],
            link["origin_slot"],
            link["target_id"],
            link["target_slot"],
            link["type"],
        )
    return tuple(link)


def assert_valid_links(graph):
    nodes = {item["id"]: item for item in graph["nodes"]}
    links = {link_values(link)[0]: link_values(link) for link in graph["links"]}

    assert len(links) == len(graph["links"])
    for link_id, origin_id, origin_slot, target_id, target_slot, link_type in links.values():
        assert origin_id in nodes
        assert target_id in nodes
        assert link_id in (nodes[origin_id]["outputs"][origin_slot].get("links") or [])
        assert nodes[target_id]["inputs"][target_slot]["link"] == link_id
        assert nodes[origin_id]["outputs"][origin_slot]["type"] == link_type
        assert nodes[target_id]["inputs"][target_slot]["type"] == link_type

    for item in nodes.values():
        for input_slot in item.get("inputs", []):
            if input_slot.get("link") is not None:
                assert input_slot["link"] in links
        for output_slot in item.get("outputs", []):
            for link_id in output_slot.get("links") or []:
                assert link_id in links


def assert_valid_workflow_links(workflow):
    assert_valid_links(workflow)
    for item in workflow["definitions"]["subgraphs"]:
        assert_valid_links(item)
```

Replace calls to `assert_valid_outer_links` with `assert_valid_workflow_links`. Add:

```python
def test_checked_in_workflows_match_builder(tmp_path):
    template_dir = Path(str(files("comfyui_workflow_templates_json").joinpath("templates")))

    build_workflows(template_dir, tmp_path)

    for name in ("quick-i2v.json", "recurring-character.json"):
        assert (tmp_path / name).read_bytes() == (WORKFLOWS / name).read_bytes()
```

Add these assertions to the existing workflow tests:

```python
assert node(workflow, "LTXVMLXTextEncode")["widgets_values"][2] == 256
```

For the recurring workflow also add:

```python
assert node(workflow, "LTXVMLXTextEncode")["widgets_values"][1] == ""
```

- [ ] **Step 2: Run workflow tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q tests-unit/ltx_stack/test_workflows.py
```

Expected: safe-default assertions fail because both max lengths are 1024 and recurring still has a negative prompt. Deterministic generation and existing links remain valid.

- [ ] **Step 3: Change only the workflow-owned defaults**

In `_quick_workflow`, change the text encoder widget tuple to:

```python
widgets=(QUICK_VIDEO_PROMPT, NEGATIVE_PROMPT, 256),
```

In `_recurring_workflow`, change it to:

```python
widgets=(RECURRING_VIDEO_PROMPT, "", 256),
```

Do not change the shared text-encoder node interface or Ingredients sampler.

- [ ] **Step 4: Regenerate checked-in workflows**

Run:

```bash
.venv/bin/python scripts/ltx_stack/build_workflows.py
```

Expected: only the text-encode widget values change in the two generated JSON files.

- [ ] **Step 5: Run workflow tests and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest -q tests-unit/ltx_stack/test_workflows.py
git diff --check
```

Expected: all workflow tests pass; no whitespace errors.

- [ ] **Step 6: Commit the workflow change**

```bash
git add scripts/ltx_stack/build_workflows.py tests-unit/ltx_stack/test_workflows.py workflows/ltx-mlx/quick-i2v.json workflows/ltx-mlx/recurring-character.json
git commit -m "Make LTX MLX workflow defaults safer"
```

### Task 3: Enforce offline startup and document real requirements

**Files:**
- Modify: `scripts/start_ltx_stack_macos.sh`
- Modify: `docs/ltx-mlx-stack.md`
- Modify: `tests-unit/ltx_stack/test_model_manifest.py`

- [ ] **Step 1: Add failing startup and guide contract tests**

Extend `test_startup_script_runs_the_local_comfy_entrypoint`:

```python
assert "export HF_HUB_OFFLINE=1" in text
```

Add:

```python
def test_operator_guide_documents_local_runtime_requirements():
    guide = (ROOT / "docs/ltx-mlx-stack.md").read_text()

    assert "64 GB" in guide
    assert "reference-front.png" in guide
    assert "reference-profile.png" in guide
    assert "local-only" in guide
    assert "f0e6f3b" in guide
    assert "HF_HUB_OFFLINE=1" in guide
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests-unit/ltx_stack/test_model_manifest.py::test_startup_script_runs_the_local_comfy_entrypoint \
  tests-unit/ltx_stack/test_model_manifest.py::test_operator_guide_documents_local_runtime_requirements
```

Expected: failures because startup is not offline and the guide omits local commit, input, and memory requirements.

- [ ] **Step 3: Make startup network-silent**

Add immediately before `cd "$ROOT_DIR"` in `scripts/start_ltx_stack_macos.sh`:

```bash
export HF_HUB_OFFLINE=1
```

Do not add download commands or network probes.

- [ ] **Step 4: Update the operator guide**

Add concise factual paragraphs covering all of the following:

```markdown
## Runtime requirements

The checked-in workflows use the `standard` MLX memory profile and target Apple Silicon systems with at least 64 GB of unified memory. On 16 GB or 32 GB systems, change both MLX loader nodes to `low_vram` before queueing. The starter prompts use a 256-token padded sequence.

This checkout is local-only until the custom-node changes through commit `f0e6f3b` are published. A stock `dgrauet/ComfyUI-LTXVideo-mlx` checkout does not contain the Ingredients node or the in-process I2V fix. Do not treat this directory as reproducible on another machine until a fork revision is published and pinned here.
```

In the workflow section, state:

```markdown
Before queueing `recurring-character.json`, upload two identity references and select them in the LoadImage nodes currently named `reference-front.png` and `reference-profile.png`. The workflow cannot run with those placeholder filenames missing.
```

In the startup section, state that the script exports `HF_HUB_OFFLINE=1` and therefore requires the verified Gemma snapshot to exist before launch. Remove any implication that `transformer-distilled-1.1.safetensors` belongs to the curated set.

- [ ] **Step 5: Run tests and shell validation**

Run:

```bash
.venv/bin/python -m pytest -q tests-unit/ltx_stack/test_model_manifest.py
bash -n scripts/start_ltx_stack_macos.sh
```

Expected: all manifest/startup/guide tests pass; shell syntax exits 0.

- [ ] **Step 6: Commit runtime documentation**

```bash
git add scripts/start_ltx_stack_macos.sh docs/ltx-mlx-stack.md tests-unit/ltx_stack/test_model_manifest.py
git commit -m "Make LTX MLX startup offline"
```

### Task 4: Verify all repositories and reclaim local storage

**Files:**
- Delete ignored symlink: `models/ltx/ltx-2.3-mlx-q8/transformer-distilled-1.1.safetensors`
- Delete cache snapshot link: `/Users/hansol/.cache/huggingface/hub/models--dgrauet--ltx-2.3-mlx-q8/snapshots/03da129baa459c9a70fc5858dee52fa417b3a93d/transformer-distilled-1.1.safetensors`
- Delete cache blob: `/Users/hansol/.cache/huggingface/hub/models--dgrauet--ltx-2.3-mlx-q8/blobs/3f5c0254c9d94abe9c913944f22bc821139010b9f1b72e65e252a50efcc74d4d`
- Delete disposable environment: `/Users/hansol/.config/superpowers/worktrees/ltx-2-mlx/comfy-progress/.venv`

- [ ] **Step 1: Run owner-scoped tests before cleanup**

Run:

```bash
.venv/bin/python -m pytest -q tests-unit/ltx_stack
/Users/hansol/dev/oss/comfy/.venv/bin/python -m pytest -q tests
```

Run the second command from `custom_nodes/ComfyUI-LTXVideo-mlx`.

Then run:

```bash
.venv/bin/python -m pytest -q tests/test_sampler_step_callback.py
```

from `/Users/hansol/.config/superpowers/worktrees/ltx-2-mlx/comfy-progress`.

Expected: stack, custom-node, and callback tests all pass.

- [ ] **Step 2: Run the full ComfyUI unit suite**

Run:

```bash
.venv/bin/python -m pytest -q tests-unit
```

Expected: no failures; current baseline is 1112 passed and 10 skipped before new tests.

- [ ] **Step 3: Verify offline startup and node registration**

Start:

```bash
./scripts/start_ltx_stack_macos.sh
```

Query the local API and assert these node types are registered:

```python
required = {
    "LTXVMLXCheckpointLoader",
    "LTXVMLXTextEncoderLoader",
    "LTXVMLXTextEncode",
    "LTXVMLXGuiderConfig",
    "LTXVMLXTwoStageSampler",
    "LTXVMLXIngredientsSampler",
}
```

Stop the diagnostic server after the API check. Expected: server starts with no custom-node import errors and all required nodes appear.

- [ ] **Step 4: Prove the large blob has no other consumers**

Run:

```bash
find /Users/hansol/.cache/huggingface/hub/models--dgrauet--ltx-2.3-mlx-q8 \
  -type l \
  -lname '*3f5c0254c9d94abe9c913944f22bc821139010b9f1b72e65e252a50efcc74d4d' \
  -print
```

Expected: exactly one snapshot symlink, the path listed in this task. Confirm no remaining manifest destination names `transformer-distilled-1.1.safetensors`.

- [ ] **Step 5: Record pre-cleanup disk usage and delete only proven-unused data**

Run:

```bash
du -sk /Users/hansol/.cache/huggingface/hub/models--dgrauet--ltx-2.3-mlx-q8
du -sk /Users/hansol/.config/superpowers/worktrees/ltx-2-mlx/comfy-progress
rm -f models/ltx/ltx-2.3-mlx-q8/transformer-distilled-1.1.safetensors
rm -f /Users/hansol/.cache/huggingface/hub/models--dgrauet--ltx-2.3-mlx-q8/snapshots/03da129baa459c9a70fc5858dee52fa417b3a93d/transformer-distilled-1.1.safetensors
rm -f /Users/hansol/.cache/huggingface/hub/models--dgrauet--ltx-2.3-mlx-q8/blobs/3f5c0254c9d94abe9c913944f22bc821139010b9f1b72e65e252a50efcc74d4d
rm -rf /Users/hansol/.config/superpowers/worktrees/ltx-2-mlx/comfy-progress/.venv
```

Do not delete any other Hugging Face blob, model symlink, worktree source, branch, commit, main `.venv`, output artifact, or optional Ingredients directory.

- [ ] **Step 6: Verify post-cleanup runtime state and measured savings**

Run:

```bash
.venv/bin/python scripts/ltx_stack/verify_install.py
du -sk /Users/hansol/.cache/huggingface/hub/models--dgrauet--ltx-2.3-mlx-q8
du -sk /Users/hansol/.config/superpowers/worktrees/ltx-2-mlx/comfy-progress
git status --short --branch
git -C custom_nodes/ComfyUI-LTXVideo-mlx status --short --branch
git -C /Users/hansol/dev/oss/ltx-2-mlx status --short --branch
git -C /Users/hansol/.config/superpowers/worktrees/ltx-2-mlx/comfy-progress status --short --branch
```

Expected: verifier exits 0 with Ingredients optional; cache usage drops by about 20.6 GB decimal; worktree usage drops by about 345 MB; no unexpected tracked changes; no repository is pushed.

- [ ] **Step 7: Record final verification in the handoff**

Report exact test counts, verifier result, registered node set, deleted paths, reclaimed KiB/GB, commit hashes, and remaining blockers. Do not claim recurring-character quality or end-to-end completion while Ingredients remains absent.
