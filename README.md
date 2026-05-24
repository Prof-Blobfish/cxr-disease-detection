# CXR Disease Classification

Train, evaluate, and tune multilabel chest X-ray classifiers with diverse image backbones. The long-term objective of this project is to build a fully trained and tuned ensemble of complementary models that outperforms any single model on the target dataset.

## Project Goal

The target end state is:

- strong single-model baselines across multiple architectures
- per-class threshold tuning using validation predictions
- repeated runs across seeds to measure stability
- an ensemble of diverse models with better ranking and thresholded classification performance than any individual backbone

In short: this repository is moving from baseline training to a tuned, multi-model ensemble pipeline.

## Current Pipeline

The codebase currently supports the following workflow:

1. Train a single model from a YAML experiment config.
2. Save checkpoints, training history, predictions, and summary artifacts.
3. Tune per-class thresholds on validation predictions.
4. Generate plots for ranking and thresholded performance analysis.
5. Run multiple configs in sequence and aggregate results into a summary CSV.

## Supported Backbones

Current model support includes:

- DenseNet121
- ResNet50
- ConvNeXt-Tiny

These provide the initial diversity needed for later ensemble construction.

## Repository Layout

```text
src/
  train.py                 single-run training entrypoint
  tune_thresholds.py       per-class threshold tuning
  plot.py                  run-level plotting and diagnostics
  experiment_runner.py     multi-config orchestration
  data.py                  dataset preparation and split handling
  datasets.py              PyTorch dataset construction
  transforms.py            train/eval augmentations
  metrics.py               ranking and thresholded metrics
  models/factory.py        model builders and parameter groups

experiments/
  *.yaml                   experiment configs

outputs/
  runs/                    per-run artifacts
  experiment_summaries/    aggregate experiment summaries
```

## Setup

### 1. Install dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
pip install python-dotenv pandas numpy matplotlib pyyaml scikit-learn tqdm
```

### 2. Configure the dataset path

Create a `.env` file in the project root:

```env
DATASET_PATH=/absolute/path/to/your/dataset
```

### 3. Expected dataset contents

The current data-loading code expects a dataset directory containing files like:

- `Data_Entry_2017.csv`
- `BBox_List_2017.csv`
- `train_val_list.txt`
- `test_list.txt`
- image files under `images_*/images/*.png`

## Running the Pipeline

### Train a single model

```bash
python src/train.py --config experiments/densenet121_v1_seed42.yaml --overwrite
```

Use `--resume` instead of `--overwrite` to continue an existing run:

```bash
python src/train.py --config experiments/densenet121_v1_seed42.yaml --resume
```

### Tune thresholds for a completed run

```bash
python src/tune_thresholds.py --run-dir outputs/runs/densenet121_v1_seed42
```

This tunes per-class decision thresholds on the validation split and evaluates both default and tuned thresholds.

### Generate plots for a completed run

```bash
python src/plot.py --run-dir outputs/runs/densenet121_v1_seed42 --top-k-pr-classes 5
```

This produces visual diagnostics such as:

- training history
- per-class AUPRC bars
- prevalence vs AUPRC scatter plots
- selected validation precision-recall curves

### Run multiple experiment configs

Edit `CONFIG_PATHS` and `TRAIN_MODE` in `src/experiment_runner.py`, then run:

```bash
python src/experiment_runner.py
```

This orchestrates training, threshold tuning, plot generation, and summary export across multiple experiment configs.

## Run Artifacts

Each run directory under `outputs/runs/<run_name>/` can contain artifacts such as:

- saved config
- model checkpoints
- training history
- validation and test predictions
- ranking metrics
- threshold-tuning outputs
- per-class metric tables
- plots
- run summary JSON

These artifacts are the basis for model comparison, threshold analysis, and eventual ensembling.

## Run Naming Convention

Each run should follow this naming pattern:

`<model>_v<version>[_<trial_change>]_seed<train_seed>`

Examples:

- `densenet121_v1_seed42`
- `densenet121_v1_lrhigh_seed42`
- `densenet121_v2_seed42`
- `convnext_tiny_v2_img224_seed42`

### Meaning

- `<model>`: model family
- `v<version>`: accepted recipe version
- `<trial_change>`: optional experimental change tag
- `seed<train_seed>`: training seed

### Practical workflow

- Start with a baseline config for each backbone.
- Make one controlled change at a time in a trial config.
- Promote successful changes into the next versioned baseline.
- Treat the saved config inside each run directory as the source of truth.

## Recommended Development Path Toward the Ensemble

The most defensible path to a strong final ensemble is:

1. Establish strong baseline recipes for DenseNet121, ResNet50, and ConvNeXt-Tiny.
2. Run multi-seed sweeps for each accepted recipe.
3. Tune per-class thresholds for every strong run.
4. Compare models using both ranking metrics and thresholded operating-point metrics.
5. Select diverse models whose errors are not highly correlated.
6. Build and evaluate an ensemble from the best complementary runs.
7. Add calibration, interpretability tooling, and new backbones only after the baseline ensemble is stable.

## Success Criteria

A successful final system should show:

- better validation and test performance than any single model
- stable results across seeds
- improved thresholded metrics after per-class tuning
- meaningful gains from combining diverse backbones into one ensemble

## Roadmap

Planned next steps include:

- multi-seed sweeps
- weighted or learned ensembling
- threshold refinement
- calibration
- Grad-CAM analysis
- integration of additional architectures such as BioViL-based models