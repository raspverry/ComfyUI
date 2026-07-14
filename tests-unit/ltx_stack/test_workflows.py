import json
from pathlib import Path


ROOT = Path(__file__).parents[2]
WORKFLOWS = ROOT / "workflows/ltx-mlx"
LTX_MODEL_DIR = "models/ltx/ltx-2.3-mlx-q8"
INGREDIENTS_LORA = "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors"
Z_IMAGE_SUBGRAPH = "f2fdebf6-dfaf-43b6-9eb2-7f70613cfdc1"
FLUX_EDIT_SUBGRAPH = "65c22b29-59aa-496b-89c6-55a603658670"


def load_workflow(name):
    return json.loads((WORKFLOWS / name).read_text())


def node(workflow, node_type):
    return next(item for item in workflow["nodes"] if item["type"] == node_type)


def subgraph(workflow, subgraph_id):
    return next(item for item in workflow["definitions"]["subgraphs"] if item["id"] == subgraph_id)


def assert_valid_outer_links(workflow):
    nodes = {item["id"]: item for item in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}

    assert len(links) == len(workflow["links"])
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


def test_quick_i2v_connects_official_z_image_to_mlx_video():
    workflow = load_workflow("quick-i2v.json")

    assert workflow["version"] == 0.4
    assert Z_IMAGE_SUBGRAPH in {item["id"] for item in workflow["definitions"]["subgraphs"]}
    assert {"SaveImage", "LTXVMLXTwoStageSampler", "CreateVideo", "SaveVideo"} <= {
        item["type"] for item in workflow["nodes"]
    }
    assert_valid_outer_links(workflow)

    checkpoint = node(workflow, "LTXVMLXCheckpointLoader")
    text_encoder = node(workflow, "LTXVMLXTextEncoderLoader")
    sampler = node(workflow, "LTXVMLXTwoStageSampler")
    assert checkpoint["widgets_values"][0] == LTX_MODEL_DIR
    assert text_encoder["widgets_values"][0] == LTX_MODEL_DIR
    assert sampler["widgets_values"][:3] == [768, 448, 121]
    assert node(workflow, "CreateVideo")["widgets_values"][0] == 24

    z_image = subgraph(workflow, Z_IMAGE_SUBGRAPH)
    image_prompt = node(z_image, "CLIPTextEncode")["widgets_values"][0]
    assert "30-year-old adult woman" in image_prompt
    assert node(z_image, "EmptySD3LatentImage")["widgets_values"][:2] == [768, 448]

    video_prompt = node(workflow, "LTXVMLXTextEncode")["widgets_values"][0].lower()
    assert "30-year-old adult woman" in video_prompt
    assert all(term in video_prompt for term in ("camera", "says", "ambient"))


def test_recurring_character_builds_bf16_reference_sheet_and_ingredients_video():
    workflow = load_workflow("recurring-character.json")

    assert workflow["version"] == 0.4
    assert len([item for item in workflow["nodes"] if item["type"] == "LoadImage"]) == 2
    assert {"RepeatImageBatch", "LTXVMLXIngredientsSampler", "CreateVideo", "SaveVideo"} <= {
        item["type"] for item in workflow["nodes"]
    }
    assert_valid_outer_links(workflow)

    flux = subgraph(workflow, FLUX_EDIT_SUBGRAPH)
    assert flux["name"].startswith("Image Edit (Flux.2 Klein")
    assert node(flux, "UNETLoader")["widgets_values"][0] == "flux-2-klein-4b.safetensors"
    scheduler = node(flux, "Flux2Scheduler")
    latent = node(flux, "EmptyFlux2LatentImage")
    assert scheduler["widgets_values"] == [4, 768, 448]
    assert latent["widgets_values"] == [768, 448, 1]
    assert all(item["link"] is None for item in scheduler["inputs"] + latent["inputs"])
    assert not any(item["type"] == "GetImageSize" for item in flux["nodes"])
    sheet_prompt = node(flux, "CLIPTextEncode")["widgets_values"][0].lower()
    assert "30-year-old adult woman" in sheet_prompt
    assert all(term in sheet_prompt for term in ("black background", "no text", "clean"))

    assert node(workflow, "RepeatImageBatch")["widgets_values"] == [121]
    checkpoint = node(workflow, "LTXVMLXCheckpointLoader")
    text_encoder = node(workflow, "LTXVMLXTextEncoderLoader")
    ingredients = node(workflow, "LTXVMLXIngredientsSampler")
    assert checkpoint["widgets_values"][0] == LTX_MODEL_DIR
    assert text_encoder["widgets_values"][0] == LTX_MODEL_DIR
    assert ingredients["widgets_values"] == [INGREDIENTS_LORA, 0.5, 1.0, 768, 448, 121, 42]
    assert node(workflow, "CreateVideo")["widgets_values"][0] == 24

    prompt = node(workflow, "LTXVMLXTextEncode")["widgets_values"][0]
    assert prompt.startswith("Reference sheet:")
    assert "\n\nGenerated video:" in prompt
    assert "30-year-old adult woman" in prompt
    assert "photorealistic" in prompt.lower()
    assert all(term in prompt.lower() for term in ("camera", "says", "ambient"))
