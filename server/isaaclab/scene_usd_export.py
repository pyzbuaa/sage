"""Offline combined-scene USD/USDZ export (no Isaac Sim MCP / PhysxSchema required)."""

import os

import numpy as np
from pxr import Sdf, Usd, UsdGeom, UsdShade, UsdUtils, Vt


def _bind_textured_material(stage, mesh, usd_internal_path, texture):
    if texture is None:
        return

    vts = texture["vts"]
    fts = texture["fts"]
    texture_map_path = texture["texture_map_path"]
    tex_coords = vts[fts.reshape(-1)].reshape(-1, 2)

    tex_coords_prim = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    )
    tex_coords_prim.Set(Vt.Vec2fArray.FromNumpy(tex_coords))

    usd_mat_path = usd_internal_path + "_mat"
    material = UsdShade.Material.Define(stage, usd_mat_path)
    st_input = material.CreateInput("frame:stPrimvarName", Sdf.ValueTypeNames.Token)
    st_input.Set("st")

    pbr_shader = UsdShade.Shader.Define(stage, f"{usd_mat_path}/PBRShader")
    pbr_shader.CreateIdAttr("UsdPreviewSurface")
    if "pbr_parameters" in texture:
        pbr_parameters = texture["pbr_parameters"]
        roughness = float(pbr_parameters.get("roughness", 1.0))
        metallic = float(pbr_parameters.get("metallic", 0.0))
    else:
        roughness = 1.0
        metallic = 0.0
    pbr_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    pbr_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    pbr_shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Bool).Set(True)
    material.CreateSurfaceOutput().ConnectToSource(pbr_shader.ConnectableAPI(), "surface")

    st_reader = UsdShade.Shader.Define(stage, f"{usd_mat_path}/stReader")
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).ConnectToSource(st_input)

    diffuse_sampler = UsdShade.Shader.Define(stage, f"{usd_mat_path}/diffuseTexture")
    diffuse_sampler.CreateIdAttr("UsdUVTexture")
    diffuse_sampler.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(os.path.abspath(texture_map_path))
    diffuse_sampler.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        st_reader.ConnectableAPI(), "result"
    )
    diffuse_sampler.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    pbr_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        diffuse_sampler.ConnectableAPI(), "rgb"
    )

    mesh.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
    UsdShade.MaterialBindingAPI(mesh).Bind(material)


def add_mesh_to_stage(stage, usd_internal_path, verts, faces, texture=None):
    points = np.asarray(verts)
    face_indices = np.asarray(faces)
    vertex_counts = np.ones(face_indices.shape[0], dtype=np.int32) * 3

    mesh = UsdGeom.Mesh.Define(stage, usd_internal_path)
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(vertex_counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(face_indices))
    mesh.CreateExtentAttr([(-100, -100, -100), (100, 100, 100)])

    _bind_textured_material(stage, mesh, usd_internal_path, texture)
    return stage


def add_door_and_frame_to_stage(
    stage,
    usd_internal_path_door,
    usd_internal_path_door_frame,
    mesh_obj_door,
    mesh_obj_door_frame,
    texture_door,
    texture_door_frame,
):
    add_mesh_to_stage(
        stage,
        usd_internal_path_door_frame,
        mesh_obj_door_frame.vertices,
        mesh_obj_door_frame.faces,
        texture_door_frame,
    )
    add_mesh_to_stage(
        stage,
        usd_internal_path_door,
        mesh_obj_door.vertices,
        mesh_obj_door.faces,
        texture_door,
    )
    return stage


def write_combined_scene_usd(mesh_info_dict, usd_file_path):
    usdz_file_path = usd_file_path.replace(".usd", ".usdz")
    for path in (usd_file_path, usdz_file_path):
        if os.path.exists(path):
            os.remove(path)

    stage = Usd.Stage.CreateNew(usd_file_path)
    UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    door_ids = []
    door_frame_ids = []

    for mesh_id in mesh_info_dict:
        if mesh_id.startswith("door_"):
            if mesh_id.endswith("_frame"):
                door_frame_ids.append(mesh_id)
            else:
                door_ids.append(mesh_id)
            continue

        mesh_dict = mesh_info_dict[mesh_id]
        add_mesh_to_stage(
            stage,
            f"/World/{mesh_id}",
            mesh_dict["mesh"].vertices,
            mesh_dict["mesh"].faces,
            mesh_dict.get("texture"),
        )

    for door_id, door_frame_id in zip(sorted(door_ids), sorted(door_frame_ids)):
        mesh_dict_door = mesh_info_dict[door_id]
        mesh_dict_door_frame = mesh_info_dict[door_frame_id]
        add_door_and_frame_to_stage(
            stage,
            f"/World/{door_id}",
            f"/World/{door_frame_id}",
            mesh_dict_door["mesh"],
            mesh_dict_door_frame["mesh"],
            mesh_dict_door.get("texture"),
            mesh_dict_door_frame.get("texture"),
        )

    stage.Save()
    if not UsdUtils.CreateNewUsdzPackage(usd_file_path, usdz_file_path):
        raise RuntimeError(f"Failed to create USDZ package: {usdz_file_path}")

    return usd_file_path, usdz_file_path
