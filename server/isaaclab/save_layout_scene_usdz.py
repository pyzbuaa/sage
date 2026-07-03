import json
import os
import sys
import traceback

# Add the server directory to the Python path to import from server modules.
server_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
isaaclab_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, server_dir)
sys.path.insert(0, isaaclab_dir)
os.chdir(server_dir)

from constants import RESULTS_DIR
from scene_usd_export import write_combined_scene_usd
from tex_utils import export_layout_to_mesh_dict_list
from utils import dict_to_floor_plan


def export_room_layout_scene_usd_offline(scene_save_dir: str, usd_file_path: str) -> dict:
    """Export a combined scene USD/USDZ without a running Isaac Sim MCP session."""
    current_layout_id = os.path.basename(scene_save_dir)
    json_file_path = os.path.join(scene_save_dir, f"{current_layout_id}.json")
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"Layout JSON not found: {json_file_path}")

    with open(json_file_path, "r") as f:
        layout_data = json.load(f)

    floor_plan = dict_to_floor_plan(layout_data)
    mesh_info_dict = export_layout_to_mesh_dict_list(floor_plan)
    usd_path, usdz_path = write_combined_scene_usd(mesh_info_dict, usd_file_path)

    return {
        "status": "success",
        "usd_file_path": usd_path,
        "usdz_file_path": usdz_path,
        "mesh_count": len(mesh_info_dict),
    }


def export_scene_usdz(layout_id: str, use_isaac_mcp: bool = False) -> dict:
    """Export a single, fully-placed scene USD (+ self-contained USDZ).

    Unlike the *_separate variant (which writes per-object files with movable
    objects left at the local origin), this uses the combined exporter that
    bakes every mesh at its world transform into one stage.
    """
    scene_save_dir = os.path.join(RESULTS_DIR, layout_id)
    usd_file_path = os.path.join(scene_save_dir, f"{layout_id}_scene.usd")
    usdz_file_path = usd_file_path.replace(".usd", ".usdz")

    print(f"scene_save_dir: {scene_save_dir}")
    print(f"exporting combined scene to: {usd_file_path}")

    if use_isaac_mcp:
        from isaacsim.isaac_mcp.server import get_room_layout_scene_usd

        result = get_room_layout_scene_usd(scene_save_dir, usd_file_path)
        print("Isaac MCP result:", result)
    else:
        result = export_room_layout_scene_usd_offline(scene_save_dir, usd_file_path)
        print("Offline export result:", result)

    if os.path.exists(usdz_file_path):
        print(f"OK self-contained scene USDZ: {usdz_file_path}")
    else:
        print(f"WARNING: expected USDZ not found at {usdz_file_path}")
    if os.path.exists(usd_file_path):
        print(f"OK scene USD: {usd_file_path}")

    return result


if __name__ == "__main__":
    layout_id = sys.argv[1] if len(sys.argv) > 1 else "layout_d40e39d9"
    use_mcp = "--use-isaac-mcp" in sys.argv
    try:
        export_scene_usdz(layout_id, use_isaac_mcp=use_mcp)
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
