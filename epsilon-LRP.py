import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from model import Model, make_casual_mask


# ============================================================
# Basic Config
# ============================================================

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


# ============================================================
# Utils
# ============================================================

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
    """
    Conv1d expects input in (N, C, T) format.
    Convert data from (N, T, C) to (N, C, T) when necessary.
    """
    if x.ndim != 3:
        raise ValueError(f"x must be three-dimensional. Current shape: {x.shape}")

    if x.shape[-1] in [3, 5, 6, 10]:
        x = np.transpose(x, (0, 2, 1))

    return x.astype(np.float32)


def to_channel_last(x):
    """
    Convert model input to the channel-last format used by visualization functions.
    Training/model input: (N, C, T)
    Visualization input: (N, T, C)
    """
    if x.ndim != 3:
        raise ValueError(f"x must be three-dimensional. Current shape: {x.shape}")

    return np.transpose(x, (0, 2, 1)).astype(np.float32)


def encode_labels(labels):
    return np.array(
        [LABEL_TO_IDX[str(label)] for label in labels],
        dtype=np.int64
    )


def normalize_train_test(X_train, X_test, eps=1e-6):
    """
    X shape: (N, C, T)
    Normalize the left and right inputs independently using their respective training-set statistics.
    """
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True)

    X_train = (X_train - mean) / (std + eps)
    X_test = (X_test - mean) / (std + eps)

    return X_train, X_test


# ============================================================
# Dataset
# ============================================================

class GRFDataset(Dataset):
    def __init__(self, left, right, labels):
        self.left = torch.from_numpy(left).float()
        self.right = torch.from_numpy(right).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.left[idx], self.right[idx], self.labels[idx]


# ============================================================
# Train / Eval
# ============================================================

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


@torch.no_grad()
def save_test_inference_csv(model, loader, device, save_path):
    """
    Save every test sample to a CSV file during test inference.

    One CSV row corresponds to one test sample.
    The full fx/fy/fz sequences are stored as JSON-formatted list strings.

    Assumed channel order:
    channel 0 = fx = ML
    channel 1 = fy = AP
    channel 2 = fz = V
    """
    model.eval()

    FX_IDX = 0
    FY_IDX = 1
    FZ_IDX = 2

    all_true = []
    all_pred = []

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sample_index",
        "true_class",
        "predicted_class",
        "left_fx",
        "left_fy",
        "left_fz",
        "right_fx",
        "right_fy",
        "right_fz",
    ]

    sample_index = 0

    with save_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

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

            left_np = left.cpu().numpy()
            right_np = right.cpu().numpy()
            labels_np = labels.cpu().numpy()
            preds_np = preds.cpu().numpy()

            for i in range(labels_np.shape[0]):
                true_idx = int(labels_np[i])
                pred_idx = int(preds_np[i])

                writer.writerow({
                    "sample_index": sample_index,
                    "true_class": IDX_TO_LABEL[true_idx],
                    "predicted_class": IDX_TO_LABEL[pred_idx],

                    "left_fx": json.dumps(left_np[i, FX_IDX, :].tolist()),
                    "left_fy": json.dumps(left_np[i, FY_IDX, :].tolist()),
                    "left_fz": json.dumps(left_np[i, FZ_IDX, :].tolist()),

                    "right_fx": json.dumps(right_np[i, FX_IDX, :].tolist()),
                    "right_fy": json.dumps(right_np[i, FY_IDX, :].tolist()),
                    "right_fz": json.dumps(right_np[i, FZ_IDX, :].tolist()),
                })

                all_true.append(true_idx)
                all_pred.append(pred_idx)
                sample_index += 1

    return np.array(all_true), np.array(all_pred)


# ============================================================
# Metrics / Plot
# ============================================================

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


# ============================================================
# LRP Logic
# ============================================================

class LRPGaitTransformer:
    """
    LRP implementation tailored to the current Model architecture.

    Expected model structure:
    - model.sensorL_embed.conv
    - model.sensorR_embed.conv
    - model.encoder: ModuleList[Encoder]
    - model.decoder: ModuleList[Decoder]
    - model.classifier
    """

    def __init__(self, model, device, epsilon=1e-9):
        self.model = model
        self.device = device
        self.epsilon = epsilon
        self.activations = {}
        self.hooks = []

    def forward_hook(self, name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output_to_save = output[0].detach().clone()
            else:
                output_to_save = output.detach().clone()

            if isinstance(input, tuple):
                input_to_save = input[0].detach().clone()
            else:
                input_to_save = input.detach().clone()

            self.activations[name] = {
                "input": input_to_save,
                "output": output_to_save,
            }

        return hook

    def register_hooks(self):
        self.hooks = []

        self.hooks.append(
            self.model.sensorL_embed.conv.register_forward_hook(
                self.forward_hook("sensorL_conv")
            )
        )

        self.hooks.append(
            self.model.sensorR_embed.conv.register_forward_hook(
                self.forward_hook("sensorR_conv")
            )
        )

        for i, layer in enumerate(self.model.encoder):
            self.hooks.append(
                layer.self_attn.register_forward_hook(
                    self.forward_hook(f"encoder_{i}_attn")
                )
            )

            self.hooks.append(
                layer.ff.register_forward_hook(
                    self.forward_hook(f"encoder_{i}_ff")
                )
            )

        for i, layer in enumerate(self.model.decoder):
            self.hooks.append(
                layer.masked_attn.register_forward_hook(
                    self.forward_hook(f"decoder_{i}_masked_attn")
                )
            )

            self.hooks.append(
                layer.cross_attn.register_forward_hook(
                    self.forward_hook(f"decoder_{i}_cross_attn")
                )
            )

            self.hooks.append(
                layer.ff.register_forward_hook(
                    self.forward_hook(f"decoder_{i}_ff")
                )
            )

        self.hooks.append(
            self.model.classifier.register_forward_hook(
                self.forward_hook("classifier")
            )
        )

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

        self.hooks = []

    def lrp_linear(self, activation, relevance_out, weight, bias=None):
        """
        Epsilon-LRP for a linear layer.
        """
        if activation.dim() == 3:
            T, B, D_in = activation.shape
            D_out = weight.shape[0]

            activation_flat = activation.reshape(-1, D_in)
            relevance_out_flat = relevance_out.reshape(-1, D_out)

            z = torch.matmul(activation_flat, weight.t())

            if bias is not None:
                z = z + bias

            z = z + self.epsilon * torch.sign(z)
            z = torch.where(
                torch.abs(z) < self.epsilon,
                torch.ones_like(z) * self.epsilon,
                z
            )

            s = relevance_out_flat / z
            c = torch.matmul(s, weight)
            relevance_in_flat = activation_flat * c

            relevance_in = relevance_in_flat.reshape(T, B, D_in)

        else:
            z = torch.matmul(activation, weight.t())

            if bias is not None:
                z = z + bias

            z = z + self.epsilon * torch.sign(z)
            z = torch.where(
                torch.abs(z) < self.epsilon,
                torch.ones_like(z) * self.epsilon,
                z
            )

            s = relevance_out / z
            c = torch.matmul(s, weight)
            relevance_in = activation * c

        return relevance_in

    def lrp_conv1d(self, activation, relevance_out, conv_layer):
        """
        Epsilon-LRP for a Conv1d layer.
        activation: (B, C_in, T)
        relevance_out: (B, C_out, T)
        """
        weight = conv_layer.weight
        bias = conv_layer.bias
        stride = conv_layer.stride
        padding = conv_layer.padding
        dilation = conv_layer.dilation
        groups = conv_layer.groups

        z = torch.nn.functional.conv1d(
            activation,
            weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups
        )

        if relevance_out.shape[-1] != z.shape[-1]:
            min_t = min(relevance_out.shape[-1], z.shape[-1])
            relevance_out = relevance_out[..., :min_t]
            z = z[..., :min_t]

        z = z + self.epsilon * torch.sign(z)
        z = torch.where(
            torch.abs(z) < self.epsilon,
            torch.ones_like(z) * self.epsilon,
            z
        )

        s = relevance_out / z

        c = torch.nn.functional.conv_transpose1d(
            s,
            weight,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups
        )

        if c.shape[-1] != activation.shape[-1]:
            min_t = min(c.shape[-1], activation.shape[-1])
            c = c[..., :min_t]
            activation = activation[..., :min_t]

        relevance_in = activation * c

        return relevance_in

    def _ff_input_to_conv_format(self, ff_input, embed_dim):
        """
        Convert the feed-forward block input to Conv1d format: (B, D, T).

        Because layer.ff in the current model is Conv1d-based, the hooked input
        is often already in (B, D, T) format. Unconditionally treating it as
        (T, B, D) and permuting it would change (1, 256, 1) to (256, 1, 1),
        causing a Conv1d channel mismatch.
        """
        if ff_input.dim() != 3:
            raise ValueError(f"FF input must be three-dimensional. Current shape: {ff_input.shape}")

        # Already in Conv1d format: (B, D, T)
        if ff_input.shape[1] == embed_dim:
            return ff_input.contiguous(), "BDT"

        # Transformer format: (T, B, D)
        if ff_input.shape[2] == embed_dim:
            return ff_input.permute(1, 2, 0).contiguous(), "TBD"

        raise ValueError(
            f"Unable to interpret the FF input shape. shape={ff_input.shape}, "
            f"expected embed_dim={embed_dim}"
        )

    def _relevance_to_conv_format(self, R, ff_input_conv):
        """
        Convert relevance R to the same (B, D, T) format as ff_input_conv.
        R is usually in (T, B, D) format, but it may already be in (B, D, T).
        """
        if R.dim() != 3:
            raise ValueError(f"R must be three-dimensional. Current shape: {R.shape}")

        B, D, T = ff_input_conv.shape

        if R.shape == ff_input_conv.shape:
            return R.contiguous()

        if R.shape[0] == T and R.shape[1] == B and R.shape[2] == D:
            return R.permute(1, 2, 0).contiguous()

        # Expand single-step decoder relevance to match the encoder/FF temporal length when needed
        if R.shape[0] == 1 and R.shape[1] == B and R.shape[2] == D and T > 1:
            R = R.repeat(T, 1, 1) / T
            return R.permute(1, 2, 0).contiguous()

        raise ValueError(
            f"Unable to align the shape of R with the FF input shape. "
            f"R={R.shape}, ff_input_conv={ff_input_conv.shape}"
        )

    def lrp_sequential_conv(self, activation, relevance_out, sequential_module):
        """
        Backward relevance propagation through the FFN structure:
        Conv1d -> ReLU -> Conv1d.
        activation/relevance_out: (B, D, T)
        """
        layers = list(sequential_module.children())

        activations = [activation]

        with torch.no_grad():
            x = activation

            for layer in layers:
                x = layer(x)
                activations.append(x)

        R = relevance_out

        for i in range(len(layers) - 1, -1, -1):
            layer = layers[i]
            act_in = activations[i]

            if isinstance(layer, nn.Conv1d):
                R = self.lrp_conv1d(act_in, R, layer)

            elif isinstance(layer, nn.ReLU):
                R = R * (act_in > 0).float()

            else:
                pass

        return R

    def propagate_relevance_detailed(self, sensorL, sensorR, target_class):
        """
        sensorL, sensorR: (B, C, T)
        return:
        rel_L, rel_R: (B, T, C)
        """
        self.model.eval()

        if sensorL.dim() != 3 or sensorR.dim() != 3:
            raise ValueError(
                f"sensorL and sensorR must be three-dimensional. "
                f"sensorL: {sensorL.shape}, sensorR: {sensorR.shape}"
            )

        # Convert inputs from (B, T, C) to (B, C, T) when necessary
        if sensorL.shape[1] > sensorL.shape[2]:
            sensorL = sensorL.permute(0, 2, 1).contiguous()
            sensorR = sensorR.permute(0, 2, 1).contiguous()

        B, C, T = sensorL.shape

        tgt_seq = torch.zeros(
            (B, 1),
            dtype=torch.long,
            device=self.device
        )

        tgt_mask = make_casual_mask(
            tgt_seq.size(1),
            self.device
        )

        with torch.no_grad():
            seq_logits = self.model(
                sensorL,
                sensorR,
                tgt_seq,
                tgt_mask=tgt_mask
            )

            logits = seq_logits[-1]  # (B, num_classes)

        R = torch.zeros_like(logits)
        R[:, target_class] = logits[:, target_class]

        # classifier relevance
        classifier_info = self.activations.get("classifier", {})
        classifier_input = classifier_info.get("input")

        if classifier_input is not None:
            # classifier_input: (seq_len, B, D)
            if classifier_input.dim() == 3:
                classifier_input = classifier_input[-1]  # (B, D)

            R = self.lrp_linear(
                classifier_input,
                R,
                self.model.classifier.weight,
                self.model.classifier.bias
            )

            R = R.unsqueeze(0)  # (1, B, D)

        # decoder reverse
        for i in range(len(self.model.decoder) - 1, -1, -1):
            layer = self.model.decoder[i]

            ff_info = self.activations.get(f"decoder_{i}_ff", {})

            if ff_info.get("input") is not None:
                ff_input = ff_info["input"]  # (T_dec, B, D)

                if ff_input.dim() == 3:
                    embed_dim = layer.ff[0].in_channels
                    ff_input_conv, _ = self._ff_input_to_conv_format(ff_input, embed_dim)
                    R_conv = self._relevance_to_conv_format(R, ff_input_conv)

                    R_before_ff = self.lrp_sequential_conv(
                        ff_input_conv,
                        R_conv,
                        layer.ff
                    )

                    R = R_before_ff.permute(2, 0, 1).contiguous()

            # Treat masked attention and cross-attention as pass-through, as in visualize_lrp.py
            pass

        # Expand relevance to the encoder temporal length T because the decoder target length is 1
        if R.size(0) == 1:
            R = R.repeat(T, 1, 1) / T  # (T, B, D)

        # encoder reverse
        for i in range(len(self.model.encoder) - 1, -1, -1):
            layer = self.model.encoder[i]

            ff_info = self.activations.get(f"encoder_{i}_ff", {})

            if ff_info.get("input") is not None:
                ff_input = ff_info["input"]  # (T, B, D)

                if ff_input.dim() == 3:
                    embed_dim = layer.ff[0].in_channels
                    ff_input_conv, _ = self._ff_input_to_conv_format(ff_input, embed_dim)
                    R_conv = self._relevance_to_conv_format(R, ff_input_conv)

                    R_before_ff = self.lrp_sequential_conv(
                        ff_input_conv,
                        R_conv,
                        layer.ff
                    )

                    R = R_before_ff.permute(2, 0, 1).contiguous()

            # Treat self-attention as pass-through, as in visualize_lrp.py
            pass

        # R: (T, B, D) -> (B, T, D)
        R = R.permute(1, 0, 2).contiguous()

        # Split relevance equally because the left and right embeddings are combined by summation
        R_L = R / 2
        R_R = R / 2

        sensorL_conv_info = self.activations.get("sensorL_conv", {})
        sensorR_conv_info = self.activations.get("sensorR_conv", {})

        if sensorL_conv_info.get("input") is not None:
            sensorL_input = sensorL_conv_info["input"]  # (B, C, T)

            R_L_conv = self.lrp_conv1d(
                sensorL_input,
                R_L.permute(0, 2, 1).contiguous(),  # (B, D, T)
                self.model.sensorL_embed.conv
            )

            rel_L = R_L_conv.permute(0, 2, 1).contiguous()  # (B, T, C)
        else:
            rel_L = torch.zeros(B, T, C, device=self.device)

        if sensorR_conv_info.get("input") is not None:
            sensorR_input = sensorR_conv_info["input"]  # (B, C, T)

            R_R_conv = self.lrp_conv1d(
                sensorR_input,
                R_R.permute(0, 2, 1).contiguous(),
                self.model.sensorR_embed.conv
            )

            rel_R = R_R_conv.permute(0, 2, 1).contiguous()
        else:
            rel_R = torch.zeros(B, T, C, device=self.device)

        return rel_L, rel_R

    def compute_lrp(self, sensorL, sensorR, target_class):
        self.register_hooks()

        try:
            rel_L, rel_R = self.propagate_relevance_detailed(
                sensorL,
                sensorR,
                target_class
            )
        finally:
            self.remove_hooks()
            self.activations.clear()

        return rel_L, rel_R


# ============================================================
# LRP Visualization: Signal + Vertical Relevance Bands
# ============================================================

def _draw_relevance_bands(
    ax,
    rel_1d,
    x_max,
    y_min,
    y_max,
    pos_color=(1.0, 0.0, 0.0),
    neg_color=(0.0, 0.3, 1.0),
    sparse_top_pct=0.85,
    clip_pct=(5, 95),
    alpha_max=0.55,
    gamma=0.7
):
    """
    rel_1d: (T,)
    Positive relevance: red band
    Negative relevance: blue band
    """
    r = np.asarray(rel_1d, dtype=np.float32).squeeze()
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)

    if r.size == 0:
        return

    lo = np.percentile(r, clip_pct[0])
    hi = np.percentile(r, clip_pct[1])
    r_clipped = np.clip(r, lo, hi)

    rp = np.maximum(r_clipped, 0.0)
    rn = -np.minimum(r_clipped, 0.0)

    mag = np.abs(r_clipped)

    if np.max(mag) <= 1e-12:
        return

    thr = np.quantile(mag, sparse_top_pct)
    mask = mag >= thr

    def _norm(x):
        if x.max() <= 1e-12:
            return np.zeros_like(x)

        x = (x - x.min()) / (x.max() - x.min() + 1e-9)
        x = np.power(x, gamma)
        return x

    rp_n = _norm(rp) * mask
    rn_n = _norm(rn) * mask

    T = r.shape[0]

    rgba_p = np.zeros((1, T, 4), dtype=np.float32)
    rgba_n = np.zeros((1, T, 4), dtype=np.float32)

    rgba_p[..., 0] = pos_color[0]
    rgba_p[..., 1] = pos_color[1]
    rgba_p[..., 2] = pos_color[2]

    rgba_n[..., 0] = neg_color[0]
    rgba_n[..., 1] = neg_color[1]
    rgba_n[..., 2] = neg_color[2]

    rgba_p[..., 3] = np.clip(rp_n * alpha_max, 0.0, 1.0)
    rgba_n[..., 3] = np.clip(rn_n * alpha_max, 0.0, 1.0)

    ax.imshow(
        rgba_n,
        extent=[0, x_max, y_min, y_max],
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        zorder=0
    )

    ax.imshow(
        rgba_p,
        extent=[0, x_max, y_min, y_max],
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        zorder=1
    )


def prepare_relevance(R, mode="signed"):
    """
    mode:
    - signed: [-1, 1], preserve positive and negative relevance
    - pos: [0, 1], use only positive relevance
    """
    R = np.asarray(R, dtype=np.float32)
    R = np.nan_to_num(R, 0.0)

    if mode == "pos":
        R = np.maximum(R, 0)
        m = R.max() + 1e-12
        return R / m

    m = np.max(np.abs(R)) + 1e-12
    return R / m


def get_force_channel_indices(num_channels):
    """
    Determine the force-channel indices used for visualization.

    Assumed 3-channel layout:
    0 = fx = ML
    1 = fy = AP
    2 = fz = V

    Assumed 5-channel layout:
    0 = cop_ap
    1 = cop_ml
    2 = f_ap
    3 = f_ml
    4 = f_v
    """
    if num_channels >= 5:
        return {
            "ML": 3,
            "AP": 2,
            "V": 4,
        }

    if num_channels == 3:
        return {
            "ML": 0,
            "AP": 1,
            "V": 2,
        }

    raise ValueError(
        f"Unsupported number of channels: {num_channels}. "
        f"Expected data with either 3 or 5 channels."
    )


def visualize_signal_bands_side(
    X_all,
    rel_one,
    side_name,
    class_name,
    save_path
):
    """
    X_all: (N, T, C)
    rel_one: (T, C) or (1, T, C)

    3 rows x 2 columns:
    - Left: class-wide mean curve with relevance bands
    - Right: all class sample curves with relevance bands
    """
    if torch.is_tensor(rel_one):
        rel = rel_one.detach().cpu().numpy()
    else:
        rel = rel_one

    rel = np.asarray(rel)

    if rel.ndim == 3:
        rel = rel[0]

    if X_all.ndim != 3:
        raise ValueError(f"X_all must have shape (N, T, C). Current shape: {X_all.shape}")

    N, T, C = X_all.shape

    if rel.shape[0] != T and rel.shape[1] == T:
        rel = rel.transpose(1, 0)

    if rel.shape[0] != T:
        raise ValueError(
            f"The temporal length of relevance does not match X_all. "
            f"X_all T={T}, rel shape={rel.shape}"
        )

    ch_idx = get_force_channel_indices(C)
    order = ["ML", "AP", "V"]

    x = np.arange(T)

    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(13, 9),
        sharex=True
    )

    fig.suptitle(f"{class_name} — {side_name}", fontsize=16)

    for r, name in enumerate(order):
        c = ch_idx[name]

        all_curves = X_all[:, :, c]
        mean_curve = all_curves.mean(axis=0)

        rel_1d = prepare_relevance(rel[:, c], mode="signed")

        y_min = float(all_curves.min() * 1.05)
        y_max = float(all_curves.max() * 1.05)

        if abs(y_max - y_min) < 1e-9:
            y_min -= 1.0
            y_max += 1.0

        ax_mean = axes[r, 0]
        _draw_relevance_bands(
            ax_mean,
            rel_1d,
            x_max=T,
            y_min=y_min,
            y_max=y_max
        )

        ax_mean.plot(
            x,
            mean_curve,
            linewidth=3,
            zorder=2,
            color="black"
        )

        ax_mean.set_ylabel(name, fontsize=16)

        if r == 0:
            ax_mean.set_title("Mean of All Samples", fontsize=18)

        ax_all = axes[r, 1]
        _draw_relevance_bands(
            ax_all,
            rel_1d,
            x_max=T,
            y_min=y_min,
            y_max=y_max
        )

        for i in range(N):
            ax_all.plot(
                x,
                all_curves[i],
                linewidth=0.6,
                alpha=0.12,
                zorder=2,
                color="darkgray"
            )

        if r == 0:
            ax_all.set_title("All Samples", fontsize=18)

        for ax in (ax_mean, ax_all):
            ax.grid(False)
            ax.tick_params(axis="y", labelsize=12)
            ax.tick_params(axis="x", labelsize=12)
            ax.set_xlim(0, T - 1)
            ax.margins(x=0, y=0)

    axes[-1, 0].set_xlabel("100% Stance", fontsize=16)
    axes[-1, 1].set_xlabel("100% Stance", fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[SAVE] {save_path}")

        base = save_path.with_suffix("")

        # Save each subplot separately
        for r, name in enumerate(order):
            c = ch_idx[name]

            all_curves = X_all[:, :, c]
            mean_curve = all_curves.mean(axis=0)
            rel_1d = prepare_relevance(rel[:, c], mode="signed")

            y_min = float(all_curves.min() * 1.05)
            y_max = float(all_curves.max() * 1.05)

            if abs(y_max - y_min) < 1e-9:
                y_min -= 1.0
                y_max += 1.0

            for which in ["mean", "all"]:
                fig_s, ax_s = plt.subplots(figsize=(6, 3))

                _draw_relevance_bands(
                    ax_s,
                    rel_1d,
                    x_max=T,
                    y_min=y_min,
                    y_max=y_max
                )

                if which == "mean":
                    ax_s.plot(
                        x,
                        mean_curve,
                        linewidth=3,
                        zorder=2,
                        color="black"
                    )
                else:
                    for i in range(N):
                        ax_s.plot(
                            x,
                            all_curves[i],
                            linewidth=0.6,
                            alpha=0.12,
                            zorder=2,
                            color="darkgray"
                        )

                ax_s.set_xlim(0, T - 1)
                ax_s.margins(x=0, y=0)
                ax_s.set_xlabel("100% Stance", fontsize=14)
                ax_s.set_ylabel(name, fontsize=14)
                ax_s.grid(False)

                panel_path = Path(f"{base}__{side_name.lower()}_{name.lower()}_{which}.png")
                fig_s.savefig(panel_path, dpi=300, bbox_inches="tight")
                plt.close(fig_s)

                print(f"[SAVE] {panel_path}")

    plt.close(fig)

    return fig


def _average_relevance_list(relevance_list, mode="signed"):
    """
    relevance_list: list of np.ndarray, each shape (B, T, C)
    return: class mean relevance, shape (T, C)

    mode:
    - signed: average positive and negative relevance values as-is
    - positive: retain only positive relevance values before averaging
    - absolute: average the absolute relevance values
    """
    if len(relevance_list) == 0:
        return None

    R = np.concatenate(relevance_list, axis=0).astype(np.float32)
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)

    if mode == "positive":
        R = np.maximum(R, 0.0)
    elif mode == "absolute":
        R = np.abs(R)
    elif mode == "signed":
        pass
    else:
        raise ValueError(
            f"Unsupported relevance_average_mode: {mode}. "
            f"Use one of 'signed', 'positive', or 'absolute'."
        )

    return R.mean(axis=0)  # (T, C)


def run_lrp_band_analysis(
    model,
    X_test_left,
    X_test_right,
    y_test,
    device,
    save_dir=OUTPUT_DIR / "lrp_bands",
    num_samples_per_class=None,
    lrp_batch_size=16,
    relevance_average_mode="signed"
):
    """
    Save AttnLRP/LRP outputs as class-level means rather than per-sample results.

    - This is independent of the test-inference CSV export.
    - Relevance is computed for every test sample in each class.
    - Only class-mean relevance PNG and NPY files are saved; per-sample PNG files are not saved.

    X_test_left/right: (N, C, T)
    y_test: encoded label, shape (N,)

    relevance_average_mode:
    - signed: average positive and negative relevance values as-is
    - positive: average only positive relevance values
    - absolute: average absolute relevance values
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    X_left_plot = to_channel_last(X_test_left)    # (N, T, C), for signal plotting
    X_right_plot = to_channel_last(X_test_right)  # (N, T, C), for signal plotting

    lrp = LRPGaitTransformer(model, device)

    if num_samples_per_class is not None:
        print(
            "[LRP INFO] num_samples_per_class is no longer used. "
            "The mean relevance over all test samples in each class will be saved."
        )

    mean_summary = {}

    for class_idx in range(len(VALID_CLASSES)):
        class_name = IDX_TO_LABEL[class_idx]
        class_indices = np.where(y_test == class_idx)[0]

        if len(class_indices) == 0:
            print(f"[LRP WARN] No test samples for class: {class_name}")
            continue

        cls_mask = y_test == class_idx
        X_all_left_cls = X_left_plot[cls_mask]
        X_all_right_cls = X_right_plot[cls_mask]

        rel_L_batches = []
        rel_R_batches = []

        print(
            f"[LRP MEAN] class={class_name}, "
            f"class_idx={class_idx}, num_samples={len(class_indices)}, "
            f"mode={relevance_average_mode}"
        )

        for start in range(0, len(class_indices), lrp_batch_size):
            batch_indices = class_indices[start:start + lrp_batch_size]

            sensorL = torch.tensor(
                X_test_left[batch_indices],
                dtype=torch.float32,
                device=device
            )

            sensorR = torch.tensor(
                X_test_right[batch_indices],
                dtype=torch.float32,
                device=device
            )

            rel_L, rel_R = lrp.compute_lrp(
                sensorL,
                sensorR,
                target_class=class_idx
            )

            rel_L = torch.nan_to_num(
                rel_L,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

            rel_R = torch.nan_to_num(
                rel_R,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

            rel_L_batches.append(rel_L.detach().cpu().numpy())
            rel_R_batches.append(rel_R.detach().cpu().numpy())

        rel_L_mean = _average_relevance_list(
            rel_L_batches,
            mode=relevance_average_mode
        )

        rel_R_mean = _average_relevance_list(
            rel_R_batches,
            mode=relevance_average_mode
        )

        if rel_L_mean is None or rel_R_mean is None:
            print(f"[LRP WARN] Failed to compute mean relevance for class: {class_name}")
            continue

        np.save(save_dir / f"relevance_left_{class_name}_class_mean.npy", rel_L_mean)
        np.save(save_dir / f"relevance_right_{class_name}_class_mean.npy", rel_R_mean)

        left_save_path = save_dir / f"bands_left_{class_name}_class_mean.png"
        right_save_path = save_dir / f"bands_right_{class_name}_class_mean.png"

        visualize_signal_bands_side(
            X_all_left_cls,
            rel_L_mean,
            side_name="Left",
            class_name=f"{class_name} Class Mean LRP",
            save_path=left_save_path
        )

        visualize_signal_bands_side(
            X_all_right_cls,
            rel_R_mean,
            side_name="Right",
            class_name=f"{class_name} Class Mean LRP",
            save_path=right_save_path
        )

        mean_summary[class_name] = {
            "class_idx": int(class_idx),
            "num_samples": int(len(class_indices)),
            "left_relevance_npy": str(save_dir / f"relevance_left_{class_name}_class_mean.npy"),
            "right_relevance_npy": str(save_dir / f"relevance_right_{class_name}_class_mean.npy"),
            "left_plot": str(left_save_path),
            "right_plot": str(right_save_path),
            "average_mode": relevance_average_mode,
        }

    (save_dir / "lrp_class_mean_summary.json").write_text(
        json.dumps(mean_summary, indent=2),
        encoding="utf-8"
    )

    print(f"[LRP DONE] Saved class-mean relevance band plots to: {save_dir}")


# ============================================================
# Main
# ============================================================

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

    if not np.array_equal(y_left_train, y_right_train):
        print("[WARN] y_left_train and y_right_train differ. Using y_left_train.")

    if not np.array_equal(y_left_test, y_right_test):
        print("[WARN] y_left_test and y_right_test differ. Using y_left_test.")

    train_full = GRFDataset(X_train_left, X_train_right, y_train)
    test_dataset = GRFDataset(X_test_left, X_test_right, y_test)

    val_size = max(1, int(len(train_full) * VAL_RATIO))
    train_size = len(train_full) - val_size

    train_dataset, val_dataset = random_split(
        train_full,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
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
        max_len=max_len
    ).to(device)

    total_params, trainable_params, non_trainable_params = count_parameters(model)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
        min_lr=1e-6
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
        leave=True
    )

    for epoch in epoch_pbar:
        train_loss, train_acc = run_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train=True
        )

        val_loss, val_acc = run_one_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            train=False
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
                CHECKPOINT_DIR / "best_model.pt"
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
        CHECKPOINT_DIR / "last_model.pt"
    )

    checkpoint = torch.load(
        CHECKPOINT_DIR / "best_model.pt",
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ========================================================
    # Test inference CSV
    # ========================================================

    y_true, y_pred = save_test_inference_csv(
        model,
        test_loader,
        device,
        OUTPUT_DIR / "test_inference_samples.csv"
    )

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

    # ========================================================
    # LRP relevance band visualization
    # ========================================================

    run_lrp_band_analysis(
        model=model,
        X_test_left=X_test_left,
        X_test_right=X_test_right,
        y_test=y_test,
        device=device,
        save_dir=OUTPUT_DIR / "lrp_bands",
        lrp_batch_size=16,
        relevance_average_mode="signed"
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
        "lrp_output_dir": str(OUTPUT_DIR / "lrp_bands"),
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