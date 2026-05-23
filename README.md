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