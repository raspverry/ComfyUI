#!/usr/bin/env python3
import argparse
import json
from copy import deepcopy
from importlib.resources import files
from pathlib import Path


Z_IMAGE_TEMPLATE = "image_z_image_turbo.json"
FLUX_TEMPLATE = "image_flux2_klein_image_edit_4b_distilled.json"
Z_IMAGE_SUBGRAPH = "f2fdebf6-dfaf-43b6-9eb2-7f70613cfdc1"
FLUX_EDIT_SUBGRAPH = "65c22b29-59aa-496b-89c6-55a603658670"
LTX_MODEL_DIR = "models/ltx/ltx-2.3-mlx-q8"
GEMMA_MODEL = "mlx-community/gemma-3-12b-it-4bit"
INGREDIENTS_LORA = "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors"

IMAGE_PROMPT = (
    "Photorealistic cinematic portrait of a 30-year-old adult woman with natural skin texture, "
    "shoulder-length dark hair, and a charcoal jacket, standing on a quiet Tokyo street at dusk. "
    "Soft practical lighting, realistic proportions, shallow depth of field, 35mm photography."
)
QUICK_VIDEO_PROMPT = (
    "A photorealistic video of the same 30-year-old adult woman. She turns toward the camera, "
    "smiles naturally, and brushes windblown hair from her face. The camera slowly dollies in with "
    "subtle handheld movement. She says, \"What a beautiful evening.\" Quiet city ambient audio, "
    "distant traffic, footsteps, and a light breeze."
)
SHEET_PROMPT = (
    "Create a clean photorealistic character reference sheet of the same 30-year-old adult woman "
    "shown in image 1 and image 2. Use a black background with no text. Arrange consistent full-body "
    "front, three-quarter, side, and close-up face views in separate uncluttered panels. Preserve her "
    "identity, facial structure, hair, skin tone, and charcoal jacket."
)
RECURRING_VIDEO_PROMPT = (
    "Reference sheet: A clean black-background character sheet showing the same 30-year-old adult "
    "woman from consistent front, three-quarter, side, and close-up views, with natural skin texture, "
    "shoulder-length dark hair, and a charcoal jacket.\n\n"
    "Generated video: A photorealistic cinematic video of the same 30-year-old adult woman walking "
    "through a softly lit studio. She pauses and "
    "turns toward the camera while keeping her identity and wardrobe consistent. The camera tracks "
    "beside her, then makes a slow dolly-in. She says, \"Let's begin.\" Soft footsteps, cloth movement, "
    "room tone, and quiet ambient audio are audible."
)
NEGATIVE_PROMPT = "blurry, distorted face, deformed hands, duplicate person, text, watermark, low quality"


def _load_template(template_dir, name):
    return json.loads((template_dir / name).read_text())


def _subgraph_tree(workflow, root_id):
    definitions = workflow["definitions"]["subgraphs"]
    by_id = {subgraph["id"]: subgraph for subgraph in definitions}
    required = set()
    pending = [root_id]
    while pending:
        subgraph_id = pending.pop()
        if subgraph_id in required:
            continue
        required.add(subgraph_id)
        pending.extend(
            item["type"]
            for item in by_id[subgraph_id]["nodes"]
            if item["type"] in by_id
        )
    return [deepcopy(subgraph) for subgraph in definitions if subgraph["id"] in required]


def _find_node(container, node_type):
    return next(item for item in container["nodes"] if item["type"] == node_type)


def _node(node_id, node_type, pos, size, order, inputs=(), outputs=(), widgets=()):
    return {
        "id": node_id,
        "type": node_type,
        "pos": list(pos),
        "size": list(size),
        "flags": {},
        "order": order,
        "mode": 0,
        "inputs": [
            {"name": name, "type": input_type, "link": None}
            for name, input_type in inputs
        ],
        "outputs": [
            {"name": name, "type": output_type, "links": [], "slot_index": index}
            for index, (name, output_type) in enumerate(outputs)
        ],
        "properties": {"Node name for S&R": node_type},
        "widgets_values": list(widgets),
    }


def _connect(nodes, links, origin_id, origin_slot, target_id, target_slot, link_type):
    link_id = len(links) + 1
    origin = next(item for item in nodes if item["id"] == origin_id)
    target = next(item for item in nodes if item["id"] == target_id)
    origin["outputs"][origin_slot].setdefault("links", []).append(link_id)
    target["inputs"][target_slot]["link"] = link_id
    links.append([link_id, origin_id, origin_slot, target_id, target_slot, link_type])


def _workflow(workflow_id, nodes, links, subgraphs):
    return {
        "id": workflow_id,
        "revision": 0,
        "last_node_id": max(item["id"] for item in nodes),
        "last_link_id": len(links),
        "nodes": nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {
            "ds": {"scale": 0.75, "offset": [0, 0]},
            "workflowRendererVersion": "LG",
        },
        "version": 0.4,
        "definitions": {"subgraphs": subgraphs},
    }


def _quick_workflow(template_dir):
    source = _load_template(template_dir, Z_IMAGE_TEMPLATE)
    subgraphs = _subgraph_tree(source, Z_IMAGE_SUBGRAPH)
    z_image_definition = next(item for item in subgraphs if item["id"] == Z_IMAGE_SUBGRAPH)
    _find_node(z_image_definition, "CLIPTextEncode")["widgets_values"][0] = IMAGE_PROMPT
    _find_node(z_image_definition, "EmptySD3LatentImage")["widgets_values"][:2] = [768, 448]

    z_image = deepcopy(next(item for item in source["nodes"] if item["type"] == Z_IMAGE_SUBGRAPH))
    z_image.update({"id": 1, "pos": [-1050, 120], "order": 0, "mode": 0})
    z_image["outputs"][0]["links"] = []

    nodes = [
        z_image,
        _node(2, "SaveImage", (-560, -260), (420, 360), 8, (("images", "IMAGE"),), widgets=("ltx-mlx/quick-still",)),
        _node(
            3,
            "LTXVMLXCheckpointLoader",
            (-1050, 700),
            (360, 130),
            1,
            outputs=(("model", "LTXV_MLX_MODEL"), ("vae", "LTXV_MLX_VAE")),
            widgets=(LTX_MODEL_DIR, False, "standard"),
        ),
        _node(
            4,
            "LTXVMLXTextEncoderLoader",
            (-1050, 900),
            (360, 150),
            2,
            outputs=(("text_encoder", "LTXV_MLX_TEXT_ENCODER"),),
            widgets=(LTX_MODEL_DIR, GEMMA_MODEL, False, "standard"),
        ),
        _node(
            5,
            "LTXVMLXTextEncode",
            (-600, 900),
            (430, 300),
            3,
            inputs=(("text_encoder", "LTXV_MLX_TEXT_ENCODER"),),
            outputs=(("conditioning", "LTXV_MLX_CONDITIONING"),),
            widgets=(QUICK_VIDEO_PROMPT, NEGATIVE_PROMPT, 256),
        ),
        _node(
            6,
            "LTXVMLXGuiderConfig",
            (-560, 620),
            (300, 190),
            4,
            outputs=(("guider_config", "LTXV_MLX_GUIDER_CONFIG"),),
            widgets=(3.0, 0.0, "28", 0.7, 3.0),
        ),
        _node(
            7,
            "LTXVMLXTwoStageSampler",
            (-60, 420),
            (430, 430),
            5,
            inputs=(
                ("model", "LTXV_MLX_MODEL"),
                ("conditioning", "LTXV_MLX_CONDITIONING"),
                ("vae", "LTXV_MLX_VAE"),
                ("image", "IMAGE"),
                ("guider_config", "LTXV_MLX_GUIDER_CONFIG"),
            ),
            outputs=(("video_frames", "IMAGE"), ("audio", "AUDIO")),
            widgets=(
                768,
                448,
                121,
                42,
                30,
                3,
                "transformer-dev.safetensors",
                "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
                1.0,
                False,
                0.5,
            ),
        ),
        _node(
            8,
            "CreateVideo",
            (470, 470),
            (280, 170),
            6,
            inputs=(("images", "IMAGE"), ("audio", "AUDIO")),
            outputs=(("VIDEO", "VIDEO"),),
            widgets=(24, 8),
        ),
        _node(
            9,
            "SaveVideo",
            (850, 470),
            (330, 190),
            7,
            inputs=(("video", "VIDEO"),),
            outputs=(("video", "VIDEO"),),
            widgets=("ltx-mlx/quick-i2v", "auto", "auto"),
        ),
    ]
    links = []
    _connect(nodes, links, 1, 0, 2, 0, "IMAGE")
    _connect(nodes, links, 1, 0, 7, 3, "IMAGE")
    _connect(nodes, links, 3, 0, 7, 0, "LTXV_MLX_MODEL")
    _connect(nodes, links, 3, 1, 7, 2, "LTXV_MLX_VAE")
    _connect(nodes, links, 4, 0, 5, 0, "LTXV_MLX_TEXT_ENCODER")
    _connect(nodes, links, 5, 0, 7, 1, "LTXV_MLX_CONDITIONING")
    _connect(nodes, links, 6, 0, 7, 4, "LTXV_MLX_GUIDER_CONFIG")
    _connect(nodes, links, 7, 0, 8, 0, "IMAGE")
    _connect(nodes, links, 7, 1, 8, 1, "AUDIO")
    _connect(nodes, links, 8, 0, 9, 0, "VIDEO")
    return _workflow("a5ff9be8-6a90-4f32-a736-24601d0901fd", nodes, links, subgraphs)


def _recurring_workflow(template_dir):
    source = _load_template(template_dir, FLUX_TEMPLATE)
    subgraphs = _subgraph_tree(source, FLUX_EDIT_SUBGRAPH)
    flux_definition = next(item for item in subgraphs if item["id"] == FLUX_EDIT_SUBGRAPH)
    unet = _find_node(flux_definition, "UNETLoader")
    unet["widgets_values"][0] = "flux-2-klein-4b.safetensors"
    unet["properties"]["models"] = [
        {
            "name": "flux-2-klein-4b.safetensors",
            "url": "https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/resolve/main/flux-2-klein-4b.safetensors",
            "directory": "diffusion_models",
        }
    ]
    scheduler = _find_node(flux_definition, "Flux2Scheduler")
    latent = _find_node(flux_definition, "EmptyFlux2LatentImage")
    scheduler["widgets_values"] = [4, 768, 448]
    latent["widgets_values"] = [768, 448, 1]

    image_size = _find_node(flux_definition, "GetImageSize")
    size_links = {image_size["inputs"][0]["link"]}
    for output in image_size["outputs"]:
        size_links.update(output.get("links") or [])
    flux_definition["nodes"].remove(image_size)
    flux_definition["links"] = [
        link for link in flux_definition["links"] if link["id"] not in size_links
    ]
    for item in flux_definition["nodes"]:
        for input_slot in item.get("inputs", []):
            if input_slot.get("link") in size_links:
                input_slot["link"] = None
        for output_slot in item.get("outputs", []):
            if output_slot.get("links"):
                output_slot["links"] = [
                    link_id for link_id in output_slot["links"] if link_id not in size_links
                ]
    _find_node(flux_definition, "CLIPTextEncode")["widgets_values"][0] = SHEET_PROMPT

    first_image = deepcopy(next(item for item in source["nodes"] if item["id"] == 76))
    first_image.update({"id": 1, "pos": [-1250, 80], "order": 0, "mode": 0})
    first_image["widgets_values"] = ["reference-front.png", "image"]
    first_image["outputs"][0]["links"] = []
    first_image["outputs"][1]["links"] = None

    second_image = deepcopy(next(item for item in source["nodes"] if item["id"] == 81))
    second_image.update({"id": 2, "pos": [-1250, 560], "order": 1, "mode": 0})
    second_image["widgets_values"] = ["reference-profile.png", "image"]
    second_image["outputs"][0]["links"] = []
    second_image["outputs"][1]["links"] = None

    flux = deepcopy(next(item for item in source["nodes"] if item["type"] == FLUX_EDIT_SUBGRAPH))
    flux.update({"id": 3, "pos": [-780, 280], "order": 2, "mode": 0})
    flux["inputs"][1]["link"] = None
    flux["inputs"][2]["link"] = None
    flux["outputs"][0]["links"] = []

    nodes = [
        first_image,
        second_image,
        flux,
        _node(
            4,
            "SaveImage",
            (-280, -160),
            (420, 360),
            10,
            (("images", "IMAGE"),),
            widgets=("ltx-mlx/reference-sheet",),
        ),
        _node(
            5,
            "RepeatImageBatch",
            (-280, 400),
            (300, 110),
            3,
            inputs=(("image", "IMAGE"),),
            outputs=(("IMAGE", "IMAGE"),),
            widgets=(121,),
        ),
        _node(
            6,
            "LTXVMLXCheckpointLoader",
            (-1220, 1080),
            (360, 130),
            4,
            outputs=(("model", "LTXV_MLX_MODEL"), ("vae", "LTXV_MLX_VAE")),
            widgets=(LTX_MODEL_DIR, False, "standard"),
        ),
        _node(
            7,
            "LTXVMLXTextEncoderLoader",
            (-1220, 1280),
            (360, 150),
            5,
            outputs=(("text_encoder", "LTXV_MLX_TEXT_ENCODER"),),
            widgets=(LTX_MODEL_DIR, GEMMA_MODEL, False, "standard"),
        ),
        _node(
            8,
            "LTXVMLXTextEncode",
            (-760, 1120),
            (480, 410),
            6,
            inputs=(("text_encoder", "LTXV_MLX_TEXT_ENCODER"),),
            outputs=(("conditioning", "LTXV_MLX_CONDITIONING"),),
            widgets=(RECURRING_VIDEO_PROMPT, "", 256),
        ),
        _node(
            9,
            "LTXVMLXIngredientsSampler",
            (140, 560),
            (470, 390),
            7,
            inputs=(
                ("model", "LTXV_MLX_MODEL"),
                ("conditioning", "LTXV_MLX_CONDITIONING"),
                ("vae", "LTXV_MLX_VAE"),
                ("reference_video", "IMAGE"),
            ),
            outputs=(("video_frames", "IMAGE"), ("audio", "AUDIO")),
            widgets=(INGREDIENTS_LORA, 0.5, 1.0, 768, 448, 121, 42),
        ),
        _node(
            10,
            "CreateVideo",
            (730, 610),
            (280, 170),
            8,
            inputs=(("images", "IMAGE"), ("audio", "AUDIO")),
            outputs=(("VIDEO", "VIDEO"),),
            widgets=(24, 8),
        ),
        _node(
            11,
            "SaveVideo",
            (1110, 610),
            (330, 190),
            9,
            inputs=(("video", "VIDEO"),),
            outputs=(("video", "VIDEO"),),
            widgets=("ltx-mlx/recurring-character", "auto", "auto"),
        ),
    ]
    links = []
    _connect(nodes, links, 1, 0, 3, 1, "IMAGE")
    _connect(nodes, links, 2, 0, 3, 2, "IMAGE")
    _connect(nodes, links, 3, 0, 4, 0, "IMAGE")
    _connect(nodes, links, 3, 0, 5, 0, "IMAGE")
    _connect(nodes, links, 5, 0, 9, 3, "IMAGE")
    _connect(nodes, links, 6, 0, 9, 0, "LTXV_MLX_MODEL")
    _connect(nodes, links, 6, 1, 9, 2, "LTXV_MLX_VAE")
    _connect(nodes, links, 7, 0, 8, 0, "LTXV_MLX_TEXT_ENCODER")
    _connect(nodes, links, 8, 0, 9, 1, "LTXV_MLX_CONDITIONING")
    _connect(nodes, links, 9, 0, 10, 0, "IMAGE")
    _connect(nodes, links, 9, 1, 10, 1, "AUDIO")
    _connect(nodes, links, 10, 0, 11, 0, "VIDEO")
    return _workflow("d56085bb-a248-48fb-aa94-84ee22a4019d", nodes, links, subgraphs)


def build_workflows(template_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    workflows = {
        "quick-i2v.json": _quick_workflow(template_dir),
        "recurring-character.json": _recurring_workflow(template_dir),
    }
    for name, workflow in workflows.items():
        (output_dir / name).write_bytes((json.dumps(workflow, indent=2) + "\n").encode())


def main():
    parser = argparse.ArgumentParser(description="Build the LTX MLX starter workflows")
    parser.add_argument(
        "--template-dir",
        type=Path,
        default=Path(str(files("comfyui_workflow_templates_json").joinpath("templates"))),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "workflows/ltx-mlx",
    )
    args = parser.parse_args()
    build_workflows(args.template_dir, args.output_dir)


if __name__ == "__main__":
    main()
