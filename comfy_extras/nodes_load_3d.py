import nodes
import folder_paths
import os
import uuid

import numpy as np
import torch
from typing_extensions import override
from comfy_api.latest import IO, UI, ComfyExtension, InputImpl, Types

from pathlib import Path

_SUPPORTED_MESH_FORMATS = {"glb", "obj"}


def normalize_path(path):
    return path.replace('\\', '/')


def _normalize_color_factor(value, length: int):
    # trimesh stores baseColorFactor/emissiveFactor as either uint8 (0-255) or float (0-1).
    # glTF spec values are float [0, 1]; normalize here.
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size < length:
        return None
    arr = arr[:length]
    if np.issubdtype(np.asarray(value).dtype, np.integer) or arr.max() > 1.0 + 1e-6:
        arr = arr / 255.0
    return tuple(float(x) for x in np.clip(arr, 0.0, 1.0))


def _extract_material_props(material) -> dict | None:
    if material is None:
        return None
    props: dict = {}

    bcf = getattr(material, "baseColorFactor", None)
    if bcf is not None:
        v = _normalize_color_factor(bcf, 4)
        if v is not None:
            props["base_color_factor"] = v
    ef = getattr(material, "emissiveFactor", None)
    if ef is not None:
        v = _normalize_color_factor(ef, 3)
        if v is not None:
            props["emissive_factor"] = v
    for src_attr, dst_key in (
        ("metallicFactor", "metallic_factor"),
        ("roughnessFactor", "roughness_factor"),
        ("alphaCutoff", "alpha_cutoff"),
    ):
        v = getattr(material, src_attr, None)
        if v is not None:
            props[dst_key] = float(v)
    ds = getattr(material, "doubleSided", None)
    if ds is not None:
        props["double_sided"] = bool(ds)
    am = getattr(material, "alphaMode", None)
    if am is not None:
        props["alpha_mode"] = getattr(am, "name", None) or str(am)

    if "base_color_factor" not in props:
        # SimpleMaterial.diffuse always exists and defaults to [102, 102, 102, 255]
        # (40% gray) even when the source MTL doesn't declare Kd. Compare against the
        # trimesh default to avoid silently darkening textures that only specified map_Kd.
        diffuse = getattr(material, "diffuse", None)
        if diffuse is not None:
            d_arr = np.asarray(diffuse)
            is_default = (d_arr.dtype == np.uint8 and d_arr.shape == (4,)
                          and bool(np.array_equal(d_arr, [102, 102, 102, 255])))
            if not is_default:
                v = _normalize_color_factor(diffuse, 4)
                if v is not None:
                    props["base_color_factor"] = v

    return props or None


def _file3d_to_mesh(file_3d: Types.File3D) -> Types.MESH:
    import trimesh

    fmt = (file_3d.format or "").lower()
    if fmt not in _SUPPORTED_MESH_FORMATS:
        raise ValueError(
            f"File3DToMesh only supports {sorted(_SUPPORTED_MESH_FORMATS)}, got '.{fmt}'"
        )

    source = file_3d.get_source() if file_3d.is_disk_backed else file_3d.get_data()
    loaded = trimesh.load(source, file_type=fmt, process=False)

    if isinstance(loaded, trimesh.Scene):
        geometries = [g for g in loaded.dump(concatenate=False) if isinstance(g, trimesh.Trimesh)]
        if not geometries:
            raise ValueError("File3DToMesh: scene contains no triangle meshes")
        mesh = trimesh.util.concatenate(geometries) if len(geometries) > 1 else geometries[0]
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise ValueError(f"File3DToMesh: unsupported geometry type '{type(loaded).__name__}'")

    if len(mesh.faces) == 0:
        raise ValueError("File3DToMesh: mesh has no faces (point clouds are not supported)")

    vertices = torch.from_numpy(np.ascontiguousarray(mesh.vertices, dtype=np.float32)).unsqueeze(0)
    faces = torch.from_numpy(np.ascontiguousarray(mesh.faces, dtype=np.int64)).unsqueeze(0)
    n_verts = vertices.shape[1]

    uvs = None
    vertex_colors = None
    texture = None
    material_props = None

    visual = getattr(mesh, "visual", None)
    if visual is not None:
        uv = getattr(visual, "uv", None)
        if uv is not None and len(uv) == n_verts:
            uvs = torch.from_numpy(np.ascontiguousarray(uv, dtype=np.float32)).unsqueeze(0)

        try:
            vc = getattr(visual, "vertex_colors", None)
        except (AttributeError, ValueError, KeyError):
            vc = None
        if vc is not None and len(vc) == n_verts:
            vc_arr = np.asarray(vc, dtype=np.float32) / 255.0
            if vc_arr.ndim == 2 and vc_arr.shape[1] >= 3:
                vc_arr = vc_arr[:, :4] if vc_arr.shape[1] >= 4 else vc_arr[:, :3]
                vertex_colors = torch.from_numpy(np.ascontiguousarray(vc_arr)).unsqueeze(0)

        material = getattr(visual, "material", None)
        if material is not None:
            tex_img = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
            if tex_img is not None:
                tex_np = np.asarray(tex_img.convert("RGB"), dtype=np.float32) / 255.0
                texture = torch.from_numpy(np.ascontiguousarray(tex_np)).unsqueeze(0)
            material_props = _extract_material_props(material)

    return Types.MESH(vertices, faces, uvs=uvs, vertex_colors=vertex_colors,
                      texture=texture, material_props=material_props)


class Load3D(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        input_dir = os.path.join(folder_paths.get_input_directory(), "3d")

        os.makedirs(input_dir, exist_ok=True)

        input_path = Path(input_dir)
        base_path = Path(folder_paths.get_input_directory())

        files = [
            normalize_path(str(file_path.relative_to(base_path)))
            for file_path in input_path.rglob("*")
            if file_path.suffix.lower() in {'.gltf', '.glb', '.obj', '.fbx', '.stl', '.spz', '.splat', '.ply', '.ksplat'}
        ]
        return IO.Schema(
            node_id="Load3D",
            display_name="Load 3D & Animation",
            category="3d",
            essentials_category="Basics",
            is_experimental=True,
            inputs=[
                IO.Combo.Input("model_file", options=sorted(files), upload=IO.UploadType.model),
                IO.Load3D.Input("image"),
                IO.Int.Input("width", default=1024, min=1, max=4096, step=1),
                IO.Int.Input("height", default=1024, min=1, max=4096, step=1),
            ],
            outputs=[
                IO.Image.Output(display_name="image"),
                IO.Mask.Output(display_name="mask"),
                IO.String.Output(display_name="mesh_path"),
                IO.Image.Output(display_name="normal"),
                IO.Load3DCamera.Output(display_name="camera_info"),
                IO.Video.Output(display_name="recording_video"),
                IO.File3DAny.Output(display_name="model_3d"),
            ],
        )

    @classmethod
    def execute(cls, model_file, image, **kwargs) -> IO.NodeOutput:
        image_path = folder_paths.get_annotated_filepath(image['image'])
        mask_path = folder_paths.get_annotated_filepath(image['mask'])
        normal_path = folder_paths.get_annotated_filepath(image['normal'])

        load_image_node = nodes.LoadImage()
        output_image, ignore_mask = load_image_node.load_image(image=image_path)
        ignore_image, output_mask = load_image_node.load_image(image=mask_path)
        normal_image, ignore_mask2 = load_image_node.load_image(image=normal_path)

        video = None

        if image['recording'] != "":
            recording_video_path = folder_paths.get_annotated_filepath(image['recording'])

            video = InputImpl.VideoFromFile(recording_video_path)

        file_3d = Types.File3D(folder_paths.get_annotated_filepath(model_file))
        return IO.NodeOutput(output_image, output_mask, model_file, normal_image, image['camera_info'], video, file_3d)

    process = execute  # TODO: remove


class Preview3D(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Preview3D",
            search_aliases=["view mesh", "3d viewer"],
            display_name="Preview 3D & Animation",
            category="3d",
            is_experimental=True,
            is_output_node=True,
            inputs=[
                IO.MultiType.Input(
                    IO.String.Input("model_file", default="", multiline=False),
                    types=[
                        IO.File3DGLB,
                        IO.File3DGLTF,
                        IO.File3DFBX,
                        IO.File3DOBJ,
                        IO.File3DSTL,
                        IO.File3DUSDZ,
                        IO.File3DAny,
                    ],
                    tooltip="3D model file or path string",
                ),
                IO.Load3DCamera.Input("camera_info", optional=True, advanced=True),
                IO.Image.Input("bg_image", optional=True, advanced=True),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, model_file: str | Types.File3D, **kwargs) -> IO.NodeOutput:
        if isinstance(model_file, Types.File3D):
            filename = f"preview3d_{uuid.uuid4().hex}.{model_file.format}"
            model_file.save_to(os.path.join(folder_paths.get_output_directory(), filename))
        else:
            filename = model_file
        camera_info = kwargs.get("camera_info", None)
        bg_image = kwargs.get("bg_image", None)
        return IO.NodeOutput(ui=UI.PreviewUI3D(filename, camera_info, bg_image=bg_image))

    process = execute  # TODO: remove


class File3DToMesh(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="File3DToMesh",
            display_name="File3D to Mesh",
            search_aliases=["parse 3d file", "load mesh"],
            category="3d",
            is_experimental=True,
            inputs=[
                IO.MultiType.Input(
                    IO.File3DAny.Input("file_3d"),
                    types=[IO.File3DGLB, IO.File3DOBJ],
                    tooltip="3D file to parse into a MESH (.glb or .obj only)",
                ),
            ],
            outputs=[
                IO.Mesh.Output(),
            ],
        )

    @classmethod
    def execute(cls, file_3d: Types.File3D) -> IO.NodeOutput:
        return IO.NodeOutput(_file3d_to_mesh(file_3d))


class Load3DExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            Load3D,
            Preview3D,
            File3DToMesh,
        ]


async def comfy_entrypoint() -> Load3DExtension:
    return Load3DExtension()
