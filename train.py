import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import Model, make_casual_mask


SEED = 42

PROCESSED_DIR = Path("processed_data")
CHECKPOINT_DIR = Path("checkpoints")
OUTPUT_DIR = Path("output")

CHECKPOINT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

VALID_CLASSES = [
    "HC", "H_P", "H_C", "H_F",
    "K_P", "K_F", "K_R",
    "A_F", "A_R", "A_L",
    "C_F", "C_A",
]

LABEL_TO_IDX = {label: idx for idx, label in enumerate(VALID_CLASSES)}
IDX_TO_LABEL = {idx: label for label, idx in LABEL_TO_IDX.items()}

BATCH_SIZE = 32
EPOCHS = 300
LR = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
VAL_RATIO = 0.03
NUM_WORKERS = 0

EMBED_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 4
FF_DIM = 1024
DROPOUT = 0.1
PATIENCE = 20


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data():
    X_left_train = np.load(PROCESSED_DIR / "X_left_train.npy").astype(np.float32)
    X_left_test = np.load(PROCESSED_DIR / "X_left_test.npy").astype(np.float32)

    X_right_train = np.load(PROCESSED_DIR / "X_right_train.npy").astype(np.float32)
    X_right_test = np.load(PROCESSED_DIR / "X_right_test.npy").astype(np.float32)

    y_left_train = np.load(PROCESSED_DIR / "y_left_train.npy", allow_pickle=True)
    y_left_test = np.load(PROCESSED_DIR / "y_left_test.npy", allow_pickle=True)

    y_right_train = np.load(PROCESSED_DIR / "y_right_train.npy", allow_pickle=True)
    y_right_test = np.load(PROCESSED_DIR / "y_right_test.npy", allow_pickle=True)

    return (
        X_left_train,
        X_left_test,
        X_right_train,
        X_right_test,
        y_left_train,
        y_left_test,
        y_right_train,
        y_right_test,
    )


def ensure_channel_first(x):
    if x.ndim != 3:
        raise ValueError(f"x must be 3D, current shape: {x.shape}")

    if x.shape[-1] == 5:
        x = np.transpose(x, (0, 2, 1))

    return x.astype(np.float32)


def encode_labels(labels):
    return np.array(
        [LABEL_TO_IDX[str(label)] for label in labels],
        dtype=np.int64
    )


def normalize_train_test(X_train, X_test, eps=1e-6):
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True)

    X_train = (X_train - mean) / (std + eps)
    X_test = (X_test - mean) / (std + eps)

    return X_train, X_test


class GRFDataset(Dataset):
    def __init__(self, left, right, labels):
        self.left = torch.from_numpy(left).float()
        self.right = torch.from_numpy(right).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.left[idx], self.right[idx], self.labels[idx]


def run_one_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for left, right, labels in loader:
        left = left.to(device)
        right = right.to(device)
        labels = labels.to(device)

        tgt_seq = torch.zeros(
            (labels.size(0), 1),
            dtype=torch.long,
            device=device
        )
        tgt_mask = make_casual_mask(tgt_seq.size(1), device)

        with torch.set_grad_enabled(train):
            seq_logits = model(left, right, tgt_seq, tgt_mask=tgt_mask)
            logits = seq_logits[-1]
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        preds = logits.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)

    return total_loss / total_count, total_correct / total_count


@torch.no_grad()
def predict(model, loader, device):
    model.eval()

    all_true = []
    all_pred = []

    for left, right, labels in loader:
        left = left.to(device)
        right = right.to(device)
        labels = labels.to(device)

        tgt_seq = torch.zeros(
            (labels.size(0), 1),
            dtype=torch.long,
            device=device
        )
        tgt_mask = make_casual_mask(tgt_seq.size(1), device)

        seq_logits = model(left, right, tgt_seq, tgt_mask=tgt_mask)
        logits = seq_logits[-1]
        preds = logits.argmax(dim=1)

        all_true.extend(labels.cpu().numpy().tolist())
        all_pred.extend(preds.cpu().numpy().tolist())

    return np.array(all_true), np.array(all_pred)


def make_confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    return cm


def save_classification_report(y_true, y_pred, path):
    cm = make_confusion_matrix(y_true, y_pred, len(VALID_CLASSES))

    lines = ["class,precision,recall,f1-score,support"]

    for i, label in enumerate(VALID_CLASSES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        lines.append(
            f"{label},{precision:.4f},{recall:.4f},{f1:.4f},{support}"
        )

    accuracy = (y_true == y_pred).mean() if len(y_true) > 0 else 0.0
    lines.append(f"\naccuracy,{accuracy:.4f}")

    path.write_text("\n".join(lines), encoding="utf-8")


def plot_learning_curve(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train loss")
    plt.plot(epochs, history["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "learning_curve_loss.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="train acc")
    plt.plot(epochs, history["val_acc"], label="val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "learning_curve_acc.png", dpi=300)
    plt.close()


def plot_confusion_matrix(cm):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(np.arange(len(VALID_CLASSES)), VALID_CLASSES, rotation=45, ha="right")
    plt.yticks(np.arange(len(VALID_CLASSES)), VALID_CLASSES)
    plt.colorbar()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=300)
    plt.close()

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    return total_params, trainable_params, non_trainable_params

@torch.no_grad()
def measure_latency(model, test_dataset, device, warmup=20, repeat=100):
    model.eval()

    left, right, _ = test_dataset[0]

    left = left.unsqueeze(0).to(device)
    right = right.unsqueeze(0).to(device)

    tgt_seq = torch.zeros((1, 1), dtype=torch.long, device=device)
    tgt_mask = make_casual_mask(1, device)

    for _ in range(warmup):
        _ = model(left, right, tgt_seq, tgt_mask=tgt_mask)
        if device.type == "cuda":
            torch.cuda.synchronize()

    times = []

    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        for _ in range(repeat):
            starter.record()
            _ = model(left, right, tgt_seq, tgt_mask=tgt_mask)
            ender.record()

            torch.cuda.synchronize()
            times.append(starter.elapsed_time(ender))

    else:
        for _ in range(repeat):
            start = time.perf_counter()
            _ = model(left, right, tgt_seq, tgt_mask=tgt_mask)
            end = time.perf_counter()
            times.append((end - start) * 1000.0)

    return {
        "latency_ms_mean": float(np.mean(times)),
        "latency_ms_std": float(np.std(times)),
        "warmup": warmup,
        "repeat": repeat,
        "device": str(device),
        "batch_size": 1,
        "description": "end-to-end 1 sample inference latency",
    }


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    (
        X_left_train,
        X_left_test,
        X_right_train,
        X_right_test,
        y_left_train,
        y_left_test,
        y_right_train,
        y_right_test,
    ) = load_data()

    X_left_train = ensure_channel_first(X_left_train)
    X_left_test = ensure_channel_first(X_left_test)

    X_right_train = ensure_channel_first(X_right_train)
    X_right_test = ensure_channel_first(X_right_test)

    X_left_train, X_left_test = normalize_train_test(
        X_left_train,
        X_left_test
    )

    X_right_train, X_right_test = normalize_train_test(
        X_right_train,
        X_right_test
    )

    y_left_train = encode_labels(y_left_train)
    y_left_test = encode_labels(y_left_test)

    y_right_train = encode_labels(y_right_train)
    y_right_test = encode_labels(y_right_test)

    X_train_left = X_left_train
    X_train_right = X_right_train
    y_train = y_left_train

    X_test_left = X_left_test
    X_test_right = X_right_test
    y_test = y_left_test

    print("X_train_left:", X_train_left.shape)
    print("X_train_right:", X_train_right.shape)
    print("y_train:", y_train.shape)
    print("X_test_left:", X_test_left.shape)
    print("X_test_right:", X_test_right.shape)
    print("y_test:", y_test.shape)

    train_full = GRFDataset(X_train_left, X_train_right, y_train)
    test_dataset = GRFDataset(X_test_left, X_test_right, y_test)

    val_size = max(1, int(len(train_full) * VAL_RATIO))
    train_size = len(train_full) - val_size

    train_dataset, val_dataset = random_split(
        train_full,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    sensor_dim = X_train_left.shape[1]
    max_len = X_train_left.shape[2]

    model = Model(
        num_classes=len(VALID_CLASSES),
        embed_dim=EMBED_DIM,
        sensor_dim=sensor_dim,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM,
        dropout=DROPOUT,
        max_len=max_len,
    ).to(device)

    total_params, trainable_params, non_trainable_params = count_parameters(model)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",          
        factor=0.5,         
        patience=5,          
        min_lr=1e-6,
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = -1.0
    best_epoch = 0
    patience_count = 0

    epoch_pbar = tqdm(
        range(1, EPOCHS + 1),
        desc="Training Progress",
        position=0,
        leave=True,
    )

    for epoch in epoch_pbar:
        train_loss, train_acc = run_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train=True,
        )

        val_loss, val_acc = run_one_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            train=False,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step(val_acc)

        tqdm.write(
            f"[Epoch {epoch:03d}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_count = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                    "label_to_idx": LABEL_TO_IDX,
                    "idx_to_label": IDX_TO_LABEL,
                    "config": {
                        "embed_dim": EMBED_DIM,
                        "num_heads": NUM_HEADS,
                        "num_layers": NUM_LAYERS,
                        "ff_dim": FF_DIM,
                        "dropout": DROPOUT,
                        "sensor_dim": sensor_dim,
                        "max_len": max_len,
                    },
                },
                CHECKPOINT_DIR / "best_model.pt",
            )

        else:
            patience_count += 1

        if patience_count >= PATIENCE:
            tqdm.write(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    torch.save(
        {
            "epoch": len(history["train_loss"]),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "label_to_idx": LABEL_TO_IDX,
            "idx_to_label": IDX_TO_LABEL,
        },
        CHECKPOINT_DIR / "last_model.pt",
    )

    checkpoint = torch.load(
        CHECKPOINT_DIR / "best_model.pt",
        map_location=device
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    y_true, y_pred = predict(model, test_loader, device)
    test_acc = float((y_true == y_pred).mean())

    cm = make_confusion_matrix(y_true, y_pred, len(VALID_CLASSES))

    np.save(OUTPUT_DIR / "confusion_matrix.npy", cm)
    plot_confusion_matrix(cm)
    save_classification_report(
        y_true,
        y_pred,
        OUTPUT_DIR / "classification_report.txt"
    )
    plot_learning_curve(history)

    latency = measure_latency(model, test_dataset, device)

    (OUTPUT_DIR / "latency.json").write_text(
        json.dumps(latency, indent=2),
        encoding="utf-8"
    )

    (OUTPUT_DIR / "latency.txt").write_text(
        "\n".join([f"{k}: {v}" for k, v in latency.items()]),
        encoding="utf-8"
    )

    summary = {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "num_train": len(train_dataset),
        "num_val": len(val_dataset),
        "num_test": len(test_dataset),
        "checkpoint_best": str(CHECKPOINT_DIR / "best_model.pt"),
        "checkpoint_last": str(CHECKPOINT_DIR / "last_model.pt"),
        "output_dir": str(OUTPUT_DIR),
    }

    (OUTPUT_DIR / "history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8"
    )

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print("\nDone.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()