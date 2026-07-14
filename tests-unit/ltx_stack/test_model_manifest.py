import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[2]
MANIFEST_PATH = ROOT / "scripts/ltx_stack/model_manifest.json"
VERIFY_SCRIPT = ROOT / "scripts/ltx_stack/verify_install.py"


def load_manifest():
    return json.loads(MANIFEST_PATH.read_text())


def test_model_manifest_uses_only_selected_model_families():
    manifest = load_manifest()

    assert set(manifest) == {"z_image_turbo", "flux2_klein_4b", "ltx_2_3_q8", "ingredients"}
    assert manifest["ltx_2_3_q8"]["repo_id"] == "dgrauet/ltx-2.3-mlx-q8"


def test_manifest_records_exact_comfy_model_files():
    manifest = load_manifest()

    assert manifest["z_image_turbo"]["files"] == [
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
    assert manifest["flux2_klein_4b"]["files"] == [
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
    assert text_encoder["destination"] == "huggingface_cache"
    assert text_encoder["required"] is True
    assert text_encoder["license_url"] == "https://ai.google.dev/gemma/terms"


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
        "transformer-distilled-1.1.safetensors",
        "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
        "spatial_upscaler_x2_v1_1.safetensors",
        "spatial_upscaler_x2_v1_1_config.json",
        "vae_encoder.safetensors",
        "vae_decoder.safetensors",
        "audio_vae.safetensors",
        "vocoder.safetensors",
    ]
    assert all(file["destination"].startswith(f"{model['destination']}/") for file in model["files"])


def test_verifier_allows_missing_gated_ingredients(tmp_path):
    manifest = load_manifest()
    for model in manifest.values():
        if model["required"] and model.get("destination") != "huggingface_cache":
            for file in model["files"]:
                destination = tmp_path / file["destination"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.touch()
        for destination in model.get("shared_dependencies", []):
            shared = tmp_path / destination
            shared.parent.mkdir(parents=True, exist_ok=True)
            shared.touch()

    custom_node = tmp_path / "custom_nodes/ComfyUI-LTXVideo-mlx"
    custom_node.mkdir(parents=True)
    (custom_node / "__init__.py").touch()

    hf_cache = tmp_path / "hf-cache"
    text_encoder = manifest["ltx_2_3_q8"]["text_encoder"]
    repo_dir = f"models--{text_encoder['repo_id'].replace('/', '--')}"
    snapshot = hf_cache / repo_dir / "snapshots/test"
    snapshot.mkdir(parents=True)
    for filename in text_encoder["files"]:
        (snapshot / filename).touch()

    result = subprocess.run(
        [sys.executable, str(VERIFY_SCRIPT), "--root", str(tmp_path)],
        env={**os.environ, "HF_HUB_CACHE": str(hf_cache)},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "[OPTIONAL] ingredients" in result.stdout
    assert "authentication" in result.stdout


def test_verifier_fails_when_a_required_public_file_is_missing(tmp_path):
    result = subprocess.run(
        [sys.executable, str(VERIFY_SCRIPT), "--root", str(tmp_path)],
        env={**os.environ, "HF_HUB_CACHE": str(tmp_path / "hf-cache")},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "[MISSING] z_image_turbo" in result.stdout


def test_startup_script_runs_the_local_comfy_entrypoint():
    script = ROOT / "scripts/start_ltx_stack_macos.sh"
    text = script.read_text()

    assert os.access(script, os.X_OK)
    assert 'exec "$PYTHON" main.py --listen 127.0.0.1 --port 8188 --preview-method auto' in text
    assert "custom_nodes/ComfyUI-LTXVideo-mlx" in text
