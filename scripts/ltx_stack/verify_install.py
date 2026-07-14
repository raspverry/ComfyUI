#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from huggingface_hub.constants import HF_HUB_CACHE


MANIFEST_PATH = Path(__file__).with_name("model_manifest.json")


def find_snapshot(repo_id, filenames):
    repo_dir = Path(HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if not repo_dir.is_dir():
        return None
    for snapshot in repo_dir.iterdir():
        if snapshot.is_dir() and all((snapshot / filename).is_file() for filename in filenames):
            return snapshot
    return None


def verify_snapshot(name, model):
    snapshot = find_snapshot(model["repo_id"], model["files"])
    if snapshot is None:
        print(f"[MISSING] {name}: complete Hugging Face snapshot for {model['repo_id']}")
        return False
    print(f"[OK] {name}: {snapshot}")
    return True


def verify(root):
    manifest = json.loads(MANIFEST_PATH.read_text())
    root = Path(root).resolve()
    failed = False

    custom_node = root / "custom_nodes/ComfyUI-LTXVideo-mlx/__init__.py"
    if custom_node.is_file():
        print(f"[OK] custom_node: {custom_node.parent}")
    else:
        print(f"[MISSING] custom_node: {custom_node.parent}")
        failed = True

    for name, model in manifest.items():
        missing = [file["destination"] for file in model["files"] if not (root / file["destination"]).is_file()]
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
