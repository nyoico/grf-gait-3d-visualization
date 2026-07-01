from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data")
OUT_DIR = Path("processed_data_final")
OUT_DIR.mkdir(exist_ok=True)

SENSOR_GROUPS = {
    "left": [
        "GRF_COP_AP_PRO_left",
        "GRF_COP_ML_PRO_left",
        "GRF_F_AP_PRO_left",
        "GRF_F_ML_PRO_left",
        "GRF_F_V_PRO_left",
    ],
    "right": [
        "GRF_COP_AP_PRO_right",
        "GRF_COP_ML_PRO_right",
        "GRF_F_AP_PRO_right",
        "GRF_F_ML_PRO_right",
        "GRF_F_V_PRO_right",
    ],
}

ALL_SENSOR_FILES = [
    file_name
    for files in SENSOR_GROUPS.values()
    for file_name in files
]

METADATA_FILE = "GRF_metadata"

VALID_CLASSES = [
    "HC", "H_P", "H_C", "H_F",
    "K_P", "K_F", "K_R",
    "A_F", "A_R", "A_L",
    "C_F", "C_A",
]

SESSION_COL = "SESSION_ID"
TRIAL_COL = "TRIAL_ID"
LABEL_COL = "CLASS_LABEL_DETAILED"


def find_file(base_name):
    for ext in [".csv", ".xlsx", ".xls"]:
        path = DATA_DIR / f"{base_name}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"{base_name} 파일을 찾을 수 없습니다.")


def read_table(path):
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def load_sensor_file(file_name):
    path = find_file(file_name)
    df = read_table(path)
    df.columns = df.columns.astype(str).str.strip()

    value_cols = [
        col for col in df.columns
        if col not in ["SUBJECT_ID", SESSION_COL, TRIAL_COL]
    ]

    return df[[SESSION_COL, TRIAL_COL] + value_cols].copy(), value_cols


metadata_path = find_file(METADATA_FILE)
metadata = read_table(metadata_path)
metadata.columns = metadata.columns.astype(str).str.strip()

metadata = metadata[metadata[LABEL_COL].isin(VALID_CLASSES)].copy()

label_map = (
    metadata
    .drop_duplicates(SESSION_COL)
    .set_index(SESSION_COL)[LABEL_COL]
    .to_dict()
)

TEST_SIZE = 0.03
RANDOM_STATE = 42

session_label_df = (
    metadata
    .drop_duplicates(SESSION_COL)
    [[SESSION_COL, LABEL_COL]]
    .copy()
)

train_session_df, test_session_df = train_test_split(
    session_label_df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True,
    stratify=session_label_df[LABEL_COL]
)

train_sessions = set(train_session_df[SESSION_COL])
test_sessions = set(test_session_df[SESSION_COL])

sensor_dfs = {}
value_cols_map = {}

for file_name in ALL_SENSOR_FILES:
    df, value_cols = load_sensor_file(file_name)
    sensor_dfs[file_name] = df
    value_cols_map[file_name] = value_cols


def make_base_keys(sensor_files):
    base_df = sensor_dfs[sensor_files[0]][[SESSION_COL, TRIAL_COL]].copy()

    for file_name in sensor_files[1:]:
        base_df = base_df.merge(
            sensor_dfs[file_name][[SESSION_COL, TRIAL_COL]],
            on=[SESSION_COL, TRIAL_COL],
            how="inner",
        )

    base_df = base_df.drop_duplicates().reset_index(drop=True)
    base_df["label"] = base_df[SESSION_COL].map(label_map)

    base_df = base_df[base_df["label"].isin(VALID_CLASSES)].copy()

    return base_df


def extract_side_split(keys_df, side, split_name):
    sensor_files = SENSOR_GROUPS[side]

    X_channels = []

    for file_name in sensor_files:
        df = sensor_dfs[file_name]
        value_cols = value_cols_map[file_name]

        merged = keys_df[[SESSION_COL, TRIAL_COL]].merge(
            df,
            on=[SESSION_COL, TRIAL_COL],
            how="left",
        )

        X = merged[value_cols].to_numpy(dtype=np.float32)
        X_channels.append(X)

    X = np.stack(X_channels, axis=-1)

    y = keys_df["label"].to_numpy()
    session_ids = keys_df[SESSION_COL].to_numpy()
    trial_ids = keys_df[TRIAL_COL].to_numpy()
    sides = np.array([side] * len(keys_df))

    np.save(OUT_DIR / f"X_{side}_{split_name}.npy", X)
    np.save(OUT_DIR / f"y_{side}_{split_name}.npy", y)
    np.save(OUT_DIR / f"SESSION_ID_{side}_{split_name}.npy", session_ids)
    np.save(OUT_DIR / f"TRIAL_ID_{side}_{split_name}.npy", trial_ids)
    np.save(OUT_DIR / f"SIDE_{side}_{split_name}.npy", sides)

    print(f"{side} {split_name} X shape:", X.shape)
    print(f"{side} {split_name} y shape:", y.shape)


for side, sensor_files in SENSOR_GROUPS.items():
    base_df = make_base_keys(sensor_files)

    train_keys = base_df[base_df[SESSION_COL].isin(train_sessions)].copy()
    test_keys = base_df[base_df[SESSION_COL].isin(test_sessions)].copy()

    print(f"\n[{side}]")
    print("전체 aligned samples:", len(base_df))
    print("train samples:", len(train_keys))
    print("test samples:", len(test_keys))
    print("train sessions:", train_keys[SESSION_COL].nunique())
    print("test sessions:", test_keys[SESSION_COL].nunique())

    extract_side_split(train_keys, side, "train")
    extract_side_split(test_keys, side, "test")


print("\nPreprocessing 완료")
print(f"Saved to: {OUT_DIR}")