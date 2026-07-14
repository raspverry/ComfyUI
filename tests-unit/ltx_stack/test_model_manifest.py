import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.ltx_stack import verify_install


ROOT = Path(__file__).parents[2]
MANIFEST_PATH = ROOT / "scripts/ltx_stack/model_manifest.json"
VERIFY_SCRIPT = ROOT / "scripts/ltx_stack/verify_install.py"
CACHE_ENV_VARS = {"HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"}
EXPECTED_PACKAGE_VERSIONS = {
    "ltx-core-mlx": "0.14.18",
    "ltx-pipelines-mlx": "0.14.18",
}
CUSTOM_NODE_MAPPING_KEYS = (
    '"LTXVMLXCheckpointLoader":',
    '"LTXVMLXTextEncoderLoader":',
    '"LTXVMLXTextEncode":',
    '"LTXVMLXGuiderConfig":',
    '"LTXVMLXTwoStageSampler":',
    '"LTXVMLXIngredientsSampler":',
)
CUSTOM_NODE_IMPLEMENTATION_CAPABILITIES = (
    ("mlx_nodes/mlx_encode.py", "class LTXVMLXTextEncode"),
    ("mlx_nodes/mlx_guider.py", "class LTXVMLXGuiderConfig"),
    ("mlx_nodes/mlx_ingredients.py", "class LTXVMLXIngredientsSampler"),
)
CUSTOM_NODE_SOURCES = {
    "__init__.py": (
        "from .mlx_nodes import NODE_CLASS_MAPPINGS as MLX_NODE_CLASS_MAPPINGS\n"
        "NODE_CLASS_MAPPINGS = {}\n"
        "NODE_CLASS_MAPPINGS.update(MLX_NODE_CLASS_MAPPINGS)\n"
    ),
    "mlx_nodes/__init__.py": (
        "NODE_CLASS_MAPPINGS = {\n"
        '    "LTXVMLXCheckpointLoader": None,\n'
        '    "LTXVMLXTextEncoderLoader": None,\n'
        '    "LTXVMLXTextEncode": None,\n'
        '    "LTXVMLXGuiderConfig": None,\n'
        '    "LTXVMLXTwoStageSampler": None,\n'
        '    "LTXVMLXIngredientsSampler": None,\n'
        "}\n"
    ),
    "mlx_nodes/mlx_encode.py": "class LTXVMLXTextEncode:\n    pass\n",
    "mlx_nodes/mlx_guider.py": "class LTXVMLXGuiderConfig:\n    pass\n",
    "mlx_nodes/mlx_ingredients.py": "class LTXVMLXIngredientsSampler:\n    pass\n",
    "mlx_nodes/mlx_loader.py": (
        "class LTXVMLXCheckpointLoader:\n    pass\n\n"
        "class LTXVMLXTextEncoderLoader:\n    pass\n"
    ),
    "mlx_nodes/mlx_sampler.py": "class LTXVMLXTwoStageSampler:\n    pass\n",
    "mlx_nodes/mlx_utils.py": "def prepare_i2v_image():\n    pass\n",
}
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
EXPECTED_TEXT_ENCODER_FILES = {
    "added_tokens.json": 35,
    "chat_template.json": 1615,
    "config.json": 1141,
    "generation_config.json": 192,
    "model-00001-of-00002.safetensors": 5367455313,
    "model-00002-of-00002.safetensors": 2661219935,
    "model.safetensors.index.json": 108605,
    "preprocessor_config.json": 570,
    "processor_config.json": 70,
    "special_tokens_map.json": 662,
    "tokenizer.json": 33384568,
    "tokenizer.model": 4689074,
    "tokenizer_config.json": 1157007,
}


def load_manifest():
    return json.loads(MANIFEST_PATH.read_text())


def create_sized_file(path, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        file.truncate(size)


def create_custom_node(root):
    custom_node = root / "custom_nodes/ComfyUI-LTXVideo-mlx"
    for relative, source in CUSTOM_NODE_SOURCES.items():
        path = custom_node / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)


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
    repo = hf_cache / repo_dir
    snapshot = repo / "snapshots" / text_encoder["revision"]
    for file in text_encoder["files"]:
        create_sized_file(snapshot / file["filename"], file["size"])
    main_ref = repo / "refs/main"
    main_ref.parent.mkdir(parents=True)
    main_ref.write_text(f"{text_encoder['revision']}\n")


def create_package_metadata(root, versions=None):
    if versions is None:
        versions = EXPECTED_PACKAGE_VERSIONS
    packages = root / "packages"
    for package, version in versions.items():
        dist_info = packages / f"{package.replace('-', '_')}-{version}.dist-info"
        dist_info.mkdir(parents=True, exist_ok=True)
        (dist_info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {package}\nVersion: {version}\n"
        )
    return packages


def clean_cache_env(tmp_path, package_versions=None, **overrides):
    env = {key: value for key, value in os.environ.items() if key not in CACHE_ENV_VARS}
    env["HOME"] = str(tmp_path / "home")
    packages = create_package_metadata(tmp_path, package_versions)
    pythonpath = overrides.pop("PYTHONPATH", env.get("PYTHONPATH", ""))
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(packages), pythonpath)))
    env.update({key: str(value) for key, value in overrides.items()})
    return env


def model_locations(model):
    keys = ("filename", "destination", "repo_id")
    return [{key: file[key] for key in keys if key in file} for file in model["files"]]


def run_verifier(root, hf_cache, package_versions=None):
    return subprocess.run(
        [sys.executable, str(VERIFY_SCRIPT), "--root", str(root)],
        env=clean_cache_env(root, package_versions, HF_HUB_CACHE=hf_cache),
        capture_output=True,
        text=True,
    )


def test_model_manifest_uses_only_selected_model_families():
    manifest = load_manifest()

    assert set(manifest) == {"z_image_turbo", "flux2_klein_4b", "ltx_2_3_q8", "ingredients"}
    assert manifest["ltx_2_3_q8"]["repo_id"] == "dgrauet/ltx-2.3-mlx-q8"


def test_manifest_records_exact_comfy_model_files():
    manifest = load_manifest()

    assert model_locations(manifest["z_image_turbo"]) == [
        {
            "filename": "split_files/diffusion_models/z_image_turbo_bf16.safetensors",
            "destination": "models/diffusion_models/z_image_turbo_bf16.safetensors",
        },
        {
            "filename": "split_files/text_encoders/qwen_3_4b.safetensors",
            "destination": "models/text_encoders/qwen_3_4b.safetensors",
        },
        {
            "filename": "split_files/vae/ae.safetensors",
            "destination": "models/vae/ae.safetensors",
        },
    ]
    assert model_locations(manifest["flux2_klein_4b"]) == [
        {
            "filename": "flux-2-klein-4b.safetensors",
            "destination": "models/diffusion_models/flux-2-klein-4b.safetensors",
        },
        {
            "filename": "split_files/vae/flux2-vae.safetensors",
            "destination": "models/vae/flux2-vae.safetensors",
            "repo_id": "Comfy-Org/flux2-dev",
        },
    ]
    assert manifest["flux2_klein_4b"]["shared_dependencies"] == [
        "models/text_encoders/qwen_3_4b.safetensors"
    ]


def test_manifest_records_exact_required_file_sizes():
    manifest = load_manifest()
    sizes = {
        file["destination"]: file["size"]
        for model in manifest.values()
        if model["required"]
        for file in model["files"]
    }

    assert sizes == EXPECTED_DESTINATION_SIZES


def test_ingredients_is_the_only_optional_gated_artifact():
    manifest = load_manifest()

    for name, model in manifest.items():
        assert model["gated"] is (name == "ingredients")
        assert model["required"] is (name != "ingredients")
    assert manifest["ingredients"]["files"] == [
        {
            "filename": "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors",
            "destination": "models/loras/ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors",
        }
    ]


def test_each_model_family_has_an_https_license_source():
    for model in load_manifest().values():
        assert model["license_url"].startswith("https://")


def test_ltx_manifest_includes_its_mlx_gemma_text_encoder():
    text_encoder = load_manifest()["ltx_2_3_q8"]["text_encoder"]

    assert text_encoder["repo_id"] == "mlx-community/gemma-3-12b-it-4bit"
    assert text_encoder["revision"] == "86cc6a8dedbc456dd0e4af01a9d09f396f77e558"
    assert text_encoder["destination"] == "huggingface_cache"
    assert text_encoder["required"] is True
    assert text_encoder["license_url"] == "https://ai.google.dev/gemma/terms"
    assert {file["filename"]: file["size"] for file in text_encoder["files"]} == EXPECTED_TEXT_ENCODER_FILES


def test_ltx_q8_uses_only_the_curated_local_runtime_files():
    model = load_manifest()["ltx_2_3_q8"]
    filenames = [file["filename"] for file in model["files"]]

    assert model["destination"] == "models/ltx/ltx-2.3-mlx-q8"
    assert filenames == [
        "config.json",
        "embedded_config.json",
        "quantize_config.json",
        "split_model.json",
        "connector.safetensors",
        "transformer-dev.safetensors",
        "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
        "spatial_upscaler_x2_v1_1.safetensors",
        "spatial_upscaler_x2_v1_1_config.json",
        "vae_encoder.safetensors",
        "vae_decoder.safetensors",
        "audio_vae.safetensors",
        "vocoder.safetensors",
    ]
    assert all(file["destination"].startswith(f"{model['destination']}/") for file in model["files"])


def test_ltx_manifest_omits_unused_distilled_transformer():
    filenames = [file["filename"] for file in load_manifest()["ltx_2_3_q8"]["files"]]

    assert "transformer-distilled-1.1.safetensors" not in filenames


def test_verifier_allows_missing_gated_ingredients(tmp_path):
    hf_cache = tmp_path / "hf-cache"
    create_valid_install(tmp_path, hf_cache)

    result = run_verifier(tmp_path, hf_cache)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "[OK] ltx-core-mlx: 0.14.18" in result.stdout
    assert "[OK] ltx-pipelines-mlx: 0.14.18" in result.stdout
    assert "[OPTIONAL] ingredients" in result.stdout
    assert "authentication" in result.stdout


@pytest.mark.parametrize("cache_source", ["xdg", "legacy", "hf_over_legacy"])
def test_verifier_uses_huggingface_hub_cache_precedence(tmp_path, cache_source):
    if cache_source == "xdg":
        xdg_cache = tmp_path / "xdg"
        hf_cache = xdg_cache / "huggingface/hub"
        env = clean_cache_env(tmp_path, XDG_CACHE_HOME=xdg_cache)
    elif cache_source == "legacy":
        hf_cache = tmp_path / "legacy-cache"
        env = clean_cache_env(tmp_path, HUGGINGFACE_HUB_CACHE=hf_cache)
    else:
        hf_cache = tmp_path / "current-cache"
        env = clean_cache_env(
            tmp_path,
            HUGGINGFACE_HUB_CACHE=tmp_path / "empty-legacy-cache",
            HF_HUB_CACHE=hf_cache,
        )
    create_valid_install(tmp_path, hf_cache)

    result = subprocess.run(
        [sys.executable, str(VERIFY_SCRIPT), "--root", str(tmp_path)],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "[OK] ltx_2_3_q8.text_encoder" in result.stdout


def test_verifier_fails_when_a_required_public_file_is_missing(tmp_path):
    result = subprocess.run(
        [sys.executable, str(VERIFY_SCRIPT), "--root", str(tmp_path)],
        env=clean_cache_env(tmp_path, HF_HUB_CACHE=tmp_path / "hf-cache"),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "[MISSING] z_image_turbo" in result.stdout


def test_verifier_rejects_wrong_sized_required_file(tmp_path):
    hf_cache = tmp_path / "hf-cache"
    create_valid_install(tmp_path, hf_cache)
    (tmp_path / "models/ltx/ltx-2.3-mlx-q8/config.json").write_text("bad")

    result = run_verifier(tmp_path, hf_cache)

    assert result.returncode == 1
    assert "models/ltx/ltx-2.3-mlx-q8/config.json" in result.stdout


def test_verifier_rejects_wrong_sized_gemma_file(tmp_path):
    hf_cache = tmp_path / "hf-cache"
    create_valid_install(tmp_path, hf_cache)
    text_encoder = load_manifest()["ltx_2_3_q8"]["text_encoder"]
    repo_dir = f"models--{text_encoder['repo_id'].replace('/', '--')}"
    snapshot = hf_cache / repo_dir / "snapshots" / text_encoder["revision"]
    (snapshot / "config.json").write_text("bad")

    result = run_verifier(tmp_path, hf_cache)

    assert result.returncode == 1
    assert text_encoder["revision"] in result.stdout


@pytest.mark.parametrize("mapping_key", CUSTOM_NODE_MAPPING_KEYS)
def test_verifier_requires_custom_node_mapping_keys(tmp_path, mapping_key):
    hf_cache = tmp_path / "hf-cache"
    create_valid_install(tmp_path, hf_cache)
    mlx_init = tmp_path / "custom_nodes/ComfyUI-LTXVideo-mlx/mlx_nodes/__init__.py"
    source = mlx_init.read_text()
    assert mapping_key in source
    mlx_init.write_text(source.replace(mapping_key, mapping_key.removesuffix(":")))

    result = run_verifier(tmp_path, hf_cache)

    assert result.returncode == 1
    assert mapping_key in result.stdout


@pytest.mark.parametrize(("relative", "capability"), CUSTOM_NODE_IMPLEMENTATION_CAPABILITIES)
def test_verifier_requires_custom_node_implementation_capabilities(tmp_path, relative, capability):
    hf_cache = tmp_path / "hf-cache"
    create_valid_install(tmp_path, hf_cache)
    path = tmp_path / "custom_nodes/ComfyUI-LTXVideo-mlx" / relative
    source = path.read_text()
    assert capability in source
    path.write_text(source.replace(capability, capability.removeprefix("class ")))

    result = run_verifier(tmp_path, hf_cache)

    assert result.returncode == 1
    assert capability in result.stdout


def test_verifier_requires_custom_node_root_registration(tmp_path):
    hf_cache = tmp_path / "hf-cache"
    create_valid_install(tmp_path, hf_cache)
    (tmp_path / "custom_nodes/ComfyUI-LTXVideo-mlx/__init__.py").write_text("")

    result = run_verifier(tmp_path, hf_cache)

    assert result.returncode == 1
    assert "MLX_NODE_CLASS_MAPPINGS" in result.stdout


def test_verifier_requires_pinned_gemma_snapshot(tmp_path):
    hf_cache = tmp_path / "hf-cache"
    create_valid_install(tmp_path, hf_cache)
    text_encoder = load_manifest()["ltx_2_3_q8"]["text_encoder"]
    snapshots = hf_cache / f"models--{text_encoder['repo_id'].replace('/', '--')}" / "snapshots"
    (snapshots / text_encoder["revision"]).rename(snapshots / "wrong-revision")

    result = run_verifier(tmp_path, hf_cache)

    assert result.returncode == 1
    assert text_encoder["revision"] in result.stdout


def test_verifier_requires_gemma_main_ref_at_pinned_revision(tmp_path):
    hf_cache = tmp_path / "hf-cache"
    create_valid_install(tmp_path, hf_cache)
    text_encoder = load_manifest()["ltx_2_3_q8"]["text_encoder"]
    repo = hf_cache / f"models--{text_encoder['repo_id'].replace('/', '--')}"
    wrong_revision = "wrong-revision"
    (repo / "snapshots" / wrong_revision).mkdir()
    (repo / "refs/main").write_text(f"{wrong_revision}\n")

    result = run_verifier(tmp_path, hf_cache)

    assert result.returncode == 1
    assert text_encoder["revision"] in result.stdout


def test_verifier_rejects_wrong_ltx_package_version(monkeypatch, capsys):
    versions = {**EXPECTED_PACKAGE_VERSIONS, "ltx-pipelines-mlx": "0.0.0"}
    monkeypatch.setattr(verify_install, "version", versions.__getitem__)

    assert verify_install.verify_package_versions() is False
    output = capsys.readouterr().out
    assert "[OK] ltx-core-mlx: 0.14.18" in output
    assert "[MISMATCH] ltx-pipelines-mlx: expected 0.14.18, found 0.0.0" in output


@pytest.mark.parametrize("package", EXPECTED_PACKAGE_VERSIONS)
def test_verifier_uses_fixture_package_versions(tmp_path, package):
    hf_cache = tmp_path / "hf-cache"
    create_valid_install(tmp_path, hf_cache)
    versions = {**EXPECTED_PACKAGE_VERSIONS, package: "0.0.0"}

    result = run_verifier(tmp_path, hf_cache, versions)

    assert result.returncode == 1
    assert f"[MISMATCH] {package}: expected 0.14.18, found 0.0.0" in result.stdout


def test_startup_script_runs_the_local_comfy_entrypoint():
    script = ROOT / "scripts/start_ltx_stack_macos.sh"
    text = script.read_text()

    assert os.access(script, os.X_OK)
    assert "export HF_HUB_OFFLINE=1" in text.splitlines()
    assert 'exec "$PYTHON" main.py --listen 127.0.0.1 --port 8188 --preview-method auto' in text
    assert "custom_nodes/ComfyUI-LTXVideo-mlx" in text


def test_operator_guide_records_local_runtime_requirements():
    guide = (ROOT / "docs/ltx-mlx-stack.md").read_text()

    for requirement in (
        "64 GB",
        "reference-front.png",
        "reference-profile.png",
        "local-only",
        "f0e6f3b",
        "HF_HUB_OFFLINE=1",
    ):
        assert requirement in guide
