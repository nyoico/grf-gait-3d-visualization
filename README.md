# GRF Gait Digital Twin

This repository contains the anonymized implementation for a ground reaction force (GRF)-based gait classification and digital twin visualization framework.

The framework includes GRF preprocessing, Transformer-based gait classification, model training and evaluation, epsilon-layer-wise relevance propagation (ε-LRP) analysis, and Blender-based synchronization of gait-related outputs with a 3D digital twin environment.

![Blender Visualization](0001-0101.gif)

## Repository Structure

```text
grf-gait-digital-twin/
├── checkpoints/
│   └── best_model.pt
├── preprocessing.py
├── model.py
├── train.py
├── epsilon-LRP.py
├── blender_script.py
├── requirements.txt
└── README.md
```

## Files

* `preprocessing.py`: Preprocessing pipeline for GRF input signals.
* `model.py`: Transformer-based gait classification model.
* `train.py`: Training and evaluation script.
* `epsilon-LRP.py`: Runs model training, test evaluation, latency measurement, and class-wise ε-LRP relevance analysis.
* `blender_script.py`: Blender Python script for visualizing model outputs in a digital twin environment.
* `checkpoints/best_model.pt`: Trained model checkpoint used for inference or reproducibility checks.
* `requirements.txt`: Python package requirements.

## Installation

Create a virtual environment and install the required packages:

```bash
conda create -n gait-env -y
conda activate gait-env
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train.py
```

The training script loads the preprocessed GRF data, trains the classification model, and saves model checkpoints and evaluation outputs according to the paths configured in the script.

### Epsilon-LRP Analysis

```bash
python epsilon-LRP.py
```

The current `epsilon-LRP.py` script performs the complete experimental pipeline:

1. Loads and independently normalizes the left and right GRF inputs using training-set statistics.
2. Splits the training set into training and validation subsets.
3. Trains the Transformer classifier with early stopping.
4. Reloads `checkpoints/best_model.pt` and evaluates it on the test set.
5. Saves predictions, classification metrics, learning curves, parameter counts, and inference latency.
6. Computes class-wise ε-LRP relevance for all available test samples in each class.
7. Saves the mean left- and right-side relevance arrays and relevance-band visualizations.

By default, relevance is averaged in `signed` mode, preserving both positive and negative contributions:

- **Red bands:** positive relevance supporting the target class.
- **Blue bands:** negative relevance opposing the target class.

The relevance plots display the ML, AP, and vertical force channels. For every class, the script averages relevance over all test samples belonging to that class rather than saving one plot per sample.

The averaging behavior can be changed in `run_lrp_band_analysis()` by setting `relevance_average_mode` to one of the following values:

```text
signed    # preserve positive and negative relevance
positive  # retain only positive relevance
absolute  # average relevance magnitude
```

### ε-LRP Implementation Scope

The current implementation applies an epsilon-stabilized relevance rule to the classifier, Conv1d-based feed-forward blocks, and bilateral sensor embedding convolutions. Attention modules are treated as pass-through operations during relevance propagation. Because the left and right embeddings are added in the forward model, the propagated relevance is divided equally between the two embedding branches before being mapped back to the input channels.

This implementation is therefore tailored to the architecture defined in `model.py` and should be updated if the model structure or layer names change.

## Outputs

Running `epsilon-LRP.py` creates or updates the following files:

```text
checkpoints/
├── best_model.pt
└── last_model.pt

output/
├── classification_report.txt
├── confusion_matrix.npy
├── confusion_matrix.png
├── history.json
├── latency.json
├── latency.txt
├── learning_curve_acc.png
├── learning_curve_loss.png
├── summary.json
├── test_inference_samples.csv
└── lrp_bands/
    ├── lrp_class_mean_summary.json
    ├── relevance_left_<CLASS>_class_mean.npy
    ├── relevance_right_<CLASS>_class_mean.npy
    ├── bands_left_<CLASS>_class_mean.png
    ├── bands_right_<CLASS>_class_mean.png
    └── additional channel-level plot files
```

Classes without test samples are skipped and reported with an LRP warning in the console.

### Model Checkpoint

The trained checkpoint is provided at:

```text
checkpoints/best_model.pt
```

This checkpoint can be loaded for inference or evaluation using the model architecture defined in `model.py`.

### Blender Visualization

The Blender visualization script is provided as:

```text
blender_script.py
```

To use it, open the Blender project file, move to the **Scripting** workspace, load `blender_script.py`, and run the script inside Blender. The script synchronizes gait-related inference outputs with the digital twin visualization.
