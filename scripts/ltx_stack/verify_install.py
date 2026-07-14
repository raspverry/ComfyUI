#!/usr/bin/env python3
import argparse
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from huggingface_hub.constants import HF_HUB_CACHE


MANIFEST_PATH = Path(__file__).with_name("model_manifest.json")
PACKAGE_VERSIONS = {
    "ltx-core-mlx": "0.14.18",
    "ltx-pipelines-mlx": "0.14.18",
}
CUSTOM_NODE_CAPABILITIES = {
    "__init__.py": (
        "from .mlx_nodes import NODE_CLASS_MAPPINGS as MLX_NODE_CLASS_MAPPINGS",
        "NODE_CLASS_MAPPINGS.update(MLX_NODE_CLASS_MAPPINGS)",
    ),
    "mlx_nodes/__init__.py": (
        '"LTXVMLXCheckpointLoader":',
        '"LTXVMLXTextEncoderLoader":',
        '"LTXVMLXTextEncode":',
        '"LTXVMLXGuiderConfig":',
        '"LTXVMLXTwoStageSampler":',
        '"LTXVMLXIngredientsSampler":',
    ),
    "mlx_nodes/mlx_encode.py": ("class LTXVMLXTextEncode",),
    "mlx_nodes/mlx_guider.py": ("class LTXVMLXGuiderConfig",),
    "mlx_nodes/mlx_ingredients.py": ("class LTXVMLXIngredientsSampler",),
    "mlx_nodes/mlx_loader.py": (
        "class LTXVMLXCheckpointLoader",
        "class LTXVMLXTextEncoderLoader",
    ),
    "mlx_nodes/mlx_sampler.py": ("class LTXVMLXTwoStageSampler",),
    "mlx_nodes/mlx_utils.py": ("def prepare_i2v_image",),
}


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


def find_snapshot(model):
    repo_dir = Path(HF_HUB_CACHE) / f"models--{model['repo_id'].replace('/', '--')}"
    main_ref = repo_dir / "refs/main"
    snapshot = repo_dir / "snapshots" / model["revision"]
    if (
        main_ref.is_file()
        and main_ref.read_text().strip() == model["revision"]
        and snapshot.is_dir()
        and all(valid_file(snapshot / file["filename"], file) for file in model["files"])
    ):
        return snapshot
    return None


def verify_snapshot(name, model):
    snapshot = find_snapshot(model)
    if snapshot is None:
        print(f"[MISSING] {name}: snapshot {model['revision']} for {model['repo_id']}")
        return False
    print(f"[OK] {name}: {snapshot}")
    return True


def verify(root):
    manifest = json.loads(MANIFEST_PATH.read_text())
    root = Path(root).resolve()
    failed = not verify_custom_node(root)
    failed = not verify_package_versions() or failed

    for name, model in manifest.items():
        missing = [
            file["destination"]
            for file in model["files"]
            if not valid_file(root / file["destination"], file)
        ]
        missing.extend(path for path in model.get("shared_dependencies", []) if not (root / path).is_file())
        if not missing:
            print(f"[OK] {name}")
        elif model["required"]:
            print(f"[MISSING] {name}: {', '.join(missing)}")
            failed = True
        else:
            print(
                f"[OPTIONAL] {name}: gated artifact missing; accept the model terms and complete Hugging Face "
                f"authentication before downloading {', '.join(missing)}"
            )
        if text_encoder := model.get("text_encoder"):
            if not verify_snapshot(f"{name}.text_encoder", text_encoder):
                failed = True

    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description="Verify the local ComfyUI LTX MLX stack without network access.")
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[2], help="ComfyUI repository root")
    args = parser.parse_args()
    raise SystemExit(verify(args.root))


if __name__ == "__main__":
    main()
