from types import SimpleNamespace as NS

from click import Choice, command, option
from monai.utils.misc import set_determinism

from default_configs import *
from setup_model import process_by_case, process_by_slice


@command()
@option("--img_path", default=IMG_PATH, help="Path to CEST NIfTI volume.")
@option("--msk_path", default=MASK_PATH, help="Path to brain mask NIfTI.")
@option("--offset_path", default=OFFSET_PATH, help="Path to offset CSV file.")
@option(
    "--pos_encoder",
    type=Choice([
        None,
        "FourierFeatures",
        "MultiresEncoding",
        "LSE",
        "HashFourierEncoding",
        "FourierLorentzEncoding",
    ]),
    default=POS_ENCODER,
)
@option(
    "--activation_layer",
    type=Choice(["ReLULayer", "WIRELayer"]),
    default=ACTIVATION_LAYER,
)
@option("--slice", default=SLICE, is_flag=True)
@option(
    "--split_method",
    type=Choice(["pattern1", "pattern2", "pattern3"]),
    default=SPLIT_METHOD,
)
@option(
    "--weight_method",
    type=Choice([None, "Gaussian", "Lorentzian"]),
    default=WEIGHT_METHOD,
)
@option(
    "--lr_scheduler",
    type=Choice(["StepLR", "ExponentialLR", "ReduceLROnPlateau", "CosineAnnealingLR"]),
    default=LR_SCHEDULER,
)
@option("--optimizer", type=Choice(["Adam", "SGD"]), default=OPTIMIZER)
@option("--points_per_sample", default=POINTS_PER_SAMPLE, type=int)
@option("--slice_index", default=SLICE_INDEX, type=int)
@option("--seed", default=SEED, type=int)
@option("--log_dir", default=LOG_DIR)
@option("--output_path", default=OUTPUT_PATH, help="Output NIfTI when --all_slices is set.")
@option("--num_epochs", default=NUM_EPOCHS, type=int)
@option("--patience", default=PATIENCE, type=int)
@option("--all_slices", default=ALL_SLICES, is_flag=True)
@option("--hidden_size", default=HIDDEN_SIZE, type=int)
@option("--num_layers", default=NUM_LAYERS, type=int)
@option("--gpu", default=GPU, type=int)
@option("--lr", default=LEARNING_RATE, type=float)
@option("--nlevels", type=int, default=16)
@option("--ppm_dim", type=int, default=16)
@option("--ppm_encoding", is_flag=True, default=False)
def train(**kargs):
    args = NS(**kargs)
    set_determinism(args.seed)
    if args.all_slices:
        process_by_slice(args)
    else:
        process_by_case(args)


if __name__ == "__main__":
    train()
