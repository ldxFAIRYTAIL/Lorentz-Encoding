IMG_PATH = "../data/example/cest_volume.nii.gz"
MASK_PATH = "../data/example/brain_mask.nii.gz"
OFFSET_PATH = "../data/example/offsets_dense.csv"
OUTPUT_PATH = "output/reconstruction.nii.gz"
LOG_DIR = "runs"

POS_ENCODER = "MultiresEncoding"
ACTIVATION_LAYER = "ReLULayer"
SLICE = False
SPLIT_METHOD = "pattern2"
WEIGHT_METHOD = None
LR_SCHEDULER = "CosineAnnealingLR"
OPTIMIZER = "Adam"
NUM_EPOCHS = 10000
POINTS_PER_SAMPLE = 20480
SLICE_INDEX = 0
SEED = 42
PATIENCE = 1000
ALL_SLICES = False
HIDDEN_SIZE = 128
NUM_LAYERS = 6
GPU = 0
LEARNING_RATE = 3e-3

configs = {
    "FourierFeatures": {
        "freq_num": 64,
        "freq_scale": 1.0,
    },
    "MultiresEncoding": {
        "nlevels": None,
        "features": 3,
        "table_size": 2**18,
        "minres": (4, 4, 2),
        "maxres": (128, 104, 60),
        "ppm_encoder_kwargs": {
            "dim": None,
            "center_ranges": [
                (-4 / 50, -3 / 50),
                (-1.5 / 50, -0.5 / 50),
                (-0.5 / 50, 0.5 / 50),
                (3 / 50, 4 / 50),
            ],
            "amplitude_ranges": [(0, 0.5), (0, 0.5), (0.5, 1), (0, 0.5)],
            "gamma_ranges": [
                (4 / 50, 5 / 50),
                (25 / 50, 30 / 50),
                (1 / 50, 2 / 50),
                (3 / 50, 4 / 50),
            ],
            "clip": True,
            "ppm_encoding": True,
        },
    },
    "SIRENLayer": {
        "func": "initialize_siren_weights",
        "kargs": {
            "siner_factor": 30.0,
            "omega": 20.0,
        },
    },
    "WIRELayer": {
        "func": "initialize_wire_weights",
        "kargs": {
            "omega": 20.0,
            "sigma": 10.0,
        },
    },
}

lr_scheduler_params = {
    "ExponentialLR": {"gamma": 0.95},
    "StepLR": {"step_size": 1000, "gamma": 0.1},
    "CosineAnnealingLR": {"T_max": NUM_EPOCHS * 1.2, "eta_min": 1e-6},
    "ReduceLROnPlateau": {"mode": "min", "factor": 0.1, "patience": 100},
}
