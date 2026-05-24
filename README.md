## Install Dependencies
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132`
`pip install dotenv pandas matplotlib pyyaml scikit-learn tqdm`

Plan
    five architectures
    multi-seed sweeps
    weighted ensemble
    threshold tuning
    BioViL integration
    Grad-CAM
    calibration

train.py
    load config
    set seed
    prepare data
    build model
    build trainer
    train
    save artifacts

Model ensemble progression

    Phase 0
        DenseNet121

    Phase 1
        DenseNet121
        ConvNeXt-Tiny
        ResNet50

    Phase 2
        DenseNet121
        ConvNeXt-Tiny
        ResNet50
        BioViL-T

    Phase 3
        DenseNet121
        ConvNeXt-Small
        ResNet101
        BioViL-T

## Run Naming Convention

Each experiment run uses a descriptive name of the form:

`<model>_v<version>[_<trial_change>]_seed<train_seed>`

Examples:
- `densenet121_v1_seed42`
- `densenet121_v1_lrhigh_seed42`
- `densenet121_v2_seed42`
- `convnext_tiny_v2_img224_seed42`

Meaning:
- `<model>`: model family used for the run
- `v<version>`: current accepted recipe version
- `<trial_change>`: optional tag describing the new change being tested on top of that version
- `seed<train_seed>`: training seed for that run

Workflow:
- A baseline run uses only the model, version, and seed, such as `densenet121_v1_seed42`.
- A trial run adds the specific change being tested, such as `densenet121_v1_lrhigh_seed42`.
- If that change is adopted, the next accepted baseline increments the version, such as `densenet121_v2_seed42`.
- The full config for each run is saved in that run's output directory and serves as the source of truth for exact settings.