import bpy
import csv
import ast
from pathlib import Path


# ============================================================
# User Config
# ============================================================

TEXT_OBJECT_NAME = "GRF_text"

PRED_LABEL_OBJECT_NAME = "GRF_pred_label_text"
PRED_CLASS_OBJECT_NAME = "GRF_pred_class_text"

OLD_PRED_TEXT_OBJECT_NAME = "GRF_pred_text"

CSV_PATH = "//output/test_inference_samples.csv"

POINTS_PER_SAMPLE = 101

VALUE_DECIMALS = 4

PRED_LABEL_OFFSET = (-2.09903, -1.75333, 2.01398)

PRED_CLASS_OFFSET = (-2.09903, -1.75333, 2.01398) 

BLACK_TEXT_COLOR = (0.0, 0.0, 0.0, 1.0)
RED_TEXT_COLOR = (1.0, 0.0, 0.0, 1.0)

RED_TEXT_EMISSION_STRENGTH = 1.8


# ============================================================
# Cache / State
# ============================================================

CSV_CACHE = {
    "rows": None,
    "mtime_ns": None,
}

DISPLAY_STATE = {
    "last_frame": None,
    "last_time_idx": None,
    "sample_idx": 0,
}


# ============================================================
# Utils
# ============================================================

def resolve_path(path_string):
    return Path(bpy.path.abspath(path_string))


def load_csv_rows(path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def get_text_object():
    obj = bpy.data.objects.get(TEXT_OBJECT_NAME)

    if obj is None:
        raise RuntimeError(f"Text Object not found: {TEXT_OBJECT_NAME}")

    if obj.type != "FONT":
        raise RuntimeError(
            f"Object exists but is not a Text Object: {TEXT_OBJECT_NAME}, type={obj.type}"
        )

    return obj


def create_or_update_material(
    mat_name,
    color,
    emission_strength=0.0,
):
    mat = bpy.data.materials.get(mat_name)

    if mat is None:
        mat = bpy.data.materials.new(mat_name)
        mat.use_nodes = True

    mat.diffuse_color = color

    if mat.use_nodes:
        nodes = mat.node_tree.nodes
        bsdf = nodes.get("Principled BSDF")

        if bsdf is not None:
            if "Base Color" in bsdf.inputs:
                bsdf.inputs["Base Color"].default_value = color

            if "Emission Color" in bsdf.inputs:
                bsdf.inputs["Emission Color"].default_value = color

            if "Emission" in bsdf.inputs:
                bsdf.inputs["Emission"].default_value = color

            if "Emission Strength" in bsdf.inputs:
                bsdf.inputs["Emission Strength"].default_value = emission_strength

    return mat


def get_black_text_material():
    return create_or_update_material(
        mat_name="Predicted_Label_Black",
        color=BLACK_TEXT_COLOR,
        emission_strength=0.0,
    )


def get_red_text_material():
    return create_or_update_material(
        mat_name="Predicted_Class_Red",
        color=RED_TEXT_COLOR,
        emission_strength=RED_TEXT_EMISSION_STRENGTH,
    )


def assign_single_material(obj, mat):
    if len(obj.data.materials) == 0:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat


def create_text_object_from_main(main_obj, object_name, offset, body=""):
    new_obj = main_obj.copy()
    new_obj.data = main_obj.data.copy()

    new_obj.name = object_name
    new_obj.data.name = object_name + "_Data"
    new_obj.data.body = body

    new_obj.location.x = main_obj.location.x + offset[0]
    new_obj.location.y = main_obj.location.y + offset[1]
    new_obj.location.z = main_obj.location.z + offset[2]

    if main_obj.users_collection:
        main_obj.users_collection[0].objects.link(new_obj)
    else:
        bpy.context.collection.objects.link(new_obj)

    print(f"[GRF Text] Created Text Object: {object_name}")

    return new_obj


def get_or_create_pred_label_object():
    main_obj = get_text_object()
    obj = bpy.data.objects.get(PRED_LABEL_OBJECT_NAME)

    if obj is None:
        obj = create_text_object_from_main(
            main_obj=main_obj,
            object_name=PRED_LABEL_OBJECT_NAME,
            offset=PRED_LABEL_OFFSET,
            body="PREDICTED CLASS :",
        )

    if obj.type != "FONT":
        raise RuntimeError(
            f"Object exists but is not a Text Object: {PRED_LABEL_OBJECT_NAME}, type={obj.type}"
        )

    assign_single_material(obj, get_black_text_material())

    return obj


def get_or_create_pred_class_object():
    main_obj = get_text_object()
    obj = bpy.data.objects.get(PRED_CLASS_OBJECT_NAME)

    if obj is None:
        obj = create_text_object_from_main(
            main_obj=main_obj,
            object_name=PRED_CLASS_OBJECT_NAME,
            offset=PRED_CLASS_OFFSET,
            body="",
        )

    if obj.type != "FONT":
        raise RuntimeError(
            f"Object exists but is not a Text Object: {PRED_CLASS_OBJECT_NAME}, type={obj.type}"
        )

    assign_single_material(obj, get_red_text_material())

    return obj


def hide_old_pred_text_object():
    old_obj = bpy.data.objects.get(OLD_PRED_TEXT_OBJECT_NAME)

    if old_obj is not None:
        old_obj.hide_viewport = True
        old_obj.hide_render = True

        if old_obj.type == "FONT":
            old_obj.data.body = ""

        print(f"[GRF Text] Hidden old predicted Text Object: {OLD_PRED_TEXT_OBJECT_NAME}")


def pick_value(row, candidates, default="-"):
    for key in candidates:
        if key in row and row[key] != "":
            return row[key]
    return default


def parse_sequence_value(value):
    if value is None or value == "":
        return []

    if isinstance(value, list):
        return value

    value = str(value).strip()

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):
            return [float(x) for x in parsed]

        return [float(parsed)]

    except Exception:
        pass

    try:
        cleaned = value.replace("[", "").replace("]", "").replace(",", " ")
        parts = cleaned.split()
        return [float(x) for x in parts]
    except Exception:
        return []


SEQUENCE_COLUMN_CANDIDATES = {
    "left_fx":  ["left_fx", "left_foot_fx", "L_fx", "left_Fx"],
    "left_fy":  ["left_fy", "left_foot_fy", "L_fy", "left_Fy"],
    "left_fz":  ["left_fz", "left_foot_fz", "L_fz", "left_Fz"],

    "right_fx": ["right_fx", "right_foot_fx", "R_fx", "right_Fx"],
    "right_fy": ["right_fy", "right_foot_fy", "R_fy", "right_Fy"],
    "right_fz": ["right_fz", "right_foot_fz", "R_fz", "right_Fz"],
}


def preprocess_rows(rows):
    for row in rows:
        for canonical_name, candidates in SEQUENCE_COLUMN_CANDIDATES.items():
            raw_value = None

            for key in candidates:
                if key in row and row[key] != "":
                    raw_value = row[key]
                    break

            row[canonical_name] = parse_sequence_value(raw_value)

    return rows


def get_cached_rows(path):
    mtime_ns = path.stat().st_mtime_ns

    if CSV_CACHE["rows"] is None or CSV_CACHE["mtime_ns"] != mtime_ns:
        print("[GRF Text] Loading CSV...")

        rows = load_csv_rows(path)
        rows = preprocess_rows(rows)

        CSV_CACHE["rows"] = rows
        CSV_CACHE["mtime_ns"] = mtime_ns

        print(f"[GRF Text] Loaded {len(rows)} rows")

    return CSV_CACHE["rows"]


def get_value_at_time(row, canonical_name, time_idx, default="-"):
    seq = row.get(canonical_name, [])

    if not seq:
        return default

    time_idx = max(0, min(time_idx, len(seq) - 1))

    return f"{seq[time_idx]:.{VALUE_DECIMALS}f}"


# ============================================================
# Text Formatting
# ============================================================

def format_status_text(row, sample_idx, total_samples, time_idx):
    true_class = pick_value(row, ["true_class", "label", "class", "target_class"])

    left_fx = get_value_at_time(row, "left_fx", time_idx)
    left_fy = get_value_at_time(row, "left_fy", time_idx)
    left_fz = get_value_at_time(row, "left_fz", time_idx)

    right_fx = get_value_at_time(row, "right_fx", time_idx)
    right_fy = get_value_at_time(row, "right_fy", time_idx)
    right_fz = get_value_at_time(row, "right_fz", time_idx)

    lines = [
        f"SAMPLE {sample_idx + 1}/{total_samples}",
        f"TIME {time_idx + 1}/{POINTS_PER_SAMPLE}",
        "",
        "LEFT FOOT",
        f"fx : {left_fx}",
        f"fy : {left_fy}",
        f"fz : {left_fz}",
        "",
        "RIGHT FOOT",
        f"fx : {right_fx}",
        f"fy : {right_fy}",
        f"fz : {right_fz}",
        "",
        f"TRUE CLASS      : {true_class}",
    ]

    return "\n".join(lines)


def get_predicted_class_text(row):
    pred_class = pick_value(row, ["predicted_class", "pred_class", "prediction"])
    correct = pick_value(row, ["correct"], "")

    if correct == "1":
        result = " OK"
    elif correct == "0":
        result = " MISS"
    else:
        result = ""

    return f"{pred_class}{result}"


# ============================================================
# Frame Update
# ============================================================

def update_text_by_frame(scene):
    path = resolve_path(CSV_PATH)

    obj = get_text_object()
    pred_label_obj = get_or_create_pred_label_object()
    pred_class_obj = get_or_create_pred_class_object()

    if not path.exists():
        obj.data.body = f"Waiting for CSV:\n{path}"
        pred_label_obj.data.body = ""
        pred_class_obj.data.body = ""
        return

    try:
        rows = get_cached_rows(path)

        if not rows:
            obj.data.body = "CSV is empty"
            pred_label_obj.data.body = ""
            pred_class_obj.data.body = ""
            return

        frame = scene.frame_current

        if DISPLAY_STATE["last_frame"] == frame:
            return

        frame_offset = frame - scene.frame_start

        if frame_offset < 0:
            frame_offset = 0

        time_idx = frame_offset % POINTS_PER_SAMPLE

        last_time_idx = DISPLAY_STATE["last_time_idx"]

        if last_time_idx is not None:
            if last_time_idx == POINTS_PER_SAMPLE - 1 and time_idx == 0:
                DISPLAY_STATE["sample_idx"] += 1

                if DISPLAY_STATE["sample_idx"] >= len(rows):
                    DISPLAY_STATE["sample_idx"] = 0

        sample_idx = DISPLAY_STATE["sample_idx"]
        row = rows[sample_idx]

        obj.data.body = format_status_text(
            row=row,
            sample_idx=sample_idx,
            total_samples=len(rows),
            time_idx=time_idx,
        )

        pred_label_obj.data.body = "PREDICTED CLASS :"

        pred_class_obj.data.body = get_predicted_class_text(row)

        DISPLAY_STATE["last_frame"] = frame
        DISPLAY_STATE["last_time_idx"] = time_idx

        print(
            f"[GRF Text] frame={frame}, "
            f"sample={sample_idx + 1}, "
            f"time={time_idx + 1}, "
            f"pred={pred_class_obj.data.body}"
        )

    except Exception as exc:
        obj.data.body = f"CSV read error:\n{exc}"
        pred_label_obj.data.body = ""
        pred_class_obj.data.body = ""
        print(f"[GRF CSV Text] Error: {exc}")


# ============================================================
# Register Handler
# ============================================================

for handler in list(bpy.app.handlers.frame_change_post):
    if handler.__name__ == "grf_text_frame_handler":
        bpy.app.handlers.frame_change_post.remove(handler)


def grf_text_frame_handler(scene):
    update_text_by_frame(scene)


bpy.app.handlers.frame_change_post.append(grf_text_frame_handler)


# ============================================================
# Timeline 
# ============================================================

try:
    hide_old_pred_text_object()

    path = resolve_path(CSV_PATH)

    if path.exists():
        rows = get_cached_rows(path)

        bpy.context.scene.frame_start = 1
        bpy.context.scene.frame_end = POINTS_PER_SAMPLE

        print(
            f"[GRF Text] Timeline set: "
            f"1 ~ {POINTS_PER_SAMPLE} "
            f"(timeline is fixed to one GRF sample)"
        )

except Exception as exc:
    print(f"[GRF Text] Timeline set error: {exc}")


update_text_by_frame(bpy.context.scene)

print("[GRF Text] Fixed 101-frame GRF updater with black label and red class started")