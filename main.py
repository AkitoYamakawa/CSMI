import sys
import json
import trimesh
import numpy as np
from ai_agent import load_ai_model, interpret_json_with_ai

# ========================
# 形状生成関数
# ========================
def make_cylinder(radius, height, sections=64):
    return trimesh.creation.cylinder(radius=radius, height=height, sections=sections)

def make_cone(radius, height, sections=64):
    return trimesh.creation.cone(radius=radius, height=height, sections=sections)

def make_box(size):
    return trimesh.creation.box(extents=size)

# ========================
# メッシュ生成
# ========================
def part_to_mesh(p, offset=[0, 0, 0]):
    # JSON 内では "type" を使うのでそれを見る
    t = (p.get("type") or "").lower()
    pos_json = p.get("position", {})
    pos = np.array([
        float(pos_json.get("x", 0)) * 1000,
        float(pos_json.get("y", 0)) * 1000,
        float(pos_json.get("z", 0)) * 1000,
    ], dtype=float)

    # ---- Cylinder ----
    if t == "cylinder":
        r = float(p.get("radius", 5)) * 1000
        h = float(p.get("height", 20)) * 1000
        seg = int(p.get("segments", 64))

        mesh = make_cylinder(r, h, seg)
        mesh.apply_translation([0, 0, h / 2])
        mesh.apply_translation(pos)
        return mesh

    # ---- Box ----
    elif t == "box":
        size = [
            float(p.get("width", 5)) * 1000,
            float(p.get("depth", 5)) * 1000,
            float(p.get("height", 5)) * 1000,
        ]
        mesh = make_box(size)
        mesh.apply_translation(pos + np.array(size)/2)
        return mesh

    # ---- Silo (cylinder+cone) ----
    elif t in ["silo", "cylinder+cone"]:
        r = float(p.get("radius", 5)) * 1000
        h = float(p.get("height", 20)) * 1000

        cyl_h = h * 0.8
        cone_h = h * 0.2

        cylinder = make_cylinder(r, cyl_h)
        cylinder.apply_translation([0, 0, cyl_h / 2])

        cone = make_cone(r, cone_h)
        cone.apply_translation([0, 0, cyl_h + cone_h / 2])

        mesh = trimesh.util.concatenate([cylinder, cone])
        mesh.apply_translation(pos)
        return mesh

    # ---- Fallback ----
    else:
        mesh = make_box([1000, 1000, 1000])
        mesh.apply_translation(pos + np.array([500, 500, 500]))
        return mesh

# ========================
# メイン処理
# ========================
def main(input_json, output_stl):

    # ✔ 修正ポイント：正しい変数名 input_json を使う
    with open(input_json, "r", encoding="utf-8") as f:
        data_text = f.read()

    # モデルをロード
    model = load_ai_model()

    # JSONをAIで解釈
    interpreted = interpret_json_with_ai(model, data_text)

    meshes = []
    x_offset = 0
    for p in interpreted:
        mesh = part_to_mesh(p, offset=[x_offset, 0, 0])
        if mesh:
            meshes.append(mesh)
            x_offset += (p.get("radius_m", 5) * 2000) + 2000

    combined = trimesh.util.concatenate(meshes)
    combined.remove_duplicate_faces()
    combined.remove_degenerate_faces()
    combined.fill_holes()

    combined.export(output_stl)
    print(f"✅ Exported {output_stl} (parts: {len(meshes)}, faces: {len(combined.faces)})")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python main.py input.json output.stl")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
