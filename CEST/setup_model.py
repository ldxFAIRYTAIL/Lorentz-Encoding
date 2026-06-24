import json
import os
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader

import activation_functions
import networks
from default_configs import configs, lr_scheduler_params
from INR_dataset import get_train_test_dataset
from loss_function import WeightedMSELoss
from Trainer import Trainer
from utils import create_directories


def setup_dataloader(args, slice_index):
    dataset_split = get_train_test_dataset(
        img_path=args.img_path,
        offset_path=args.offset_path,
        mask_path=args.msk_path,
        split_method=args.split_method,
        slice=args.slice,
        points_per_sample=args.points_per_sample,
        slice_index=slice_index,
        device=f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu",
    )
    train_dataloader = DataLoader(dataset_split.train_dataset, batch_size=1, num_workers=0, pin_memory=False)
    test_dataloader = DataLoader(dataset_split.test_dataset, batch_size=1, num_workers=0, pin_memory=False)
    return dataset_split, train_dataloader, test_dataloader


def _build_multires_config(args: NS, device: str):
    config = deepcopy(configs["MultiresEncoding"])
    config["device"] = device
    config["nlevels"] = args.nlevels
    config["ppm_encoder_kwargs"]["dim"] = args.ppm_dim
    config["ppm_encoder_kwargs"]["ppm_encoding"] = args.ppm_encoding
    return config


def _build_hash_fourier_encoder(args: NS, device: str):
    spatial_config = _build_multires_config(args, device)
    spatial_config["minres"] = spatial_config["minres"][:2]
    spatial_config["maxres"] = spatial_config["maxres"][:2]
    spatial_config["nlevels"] = 16
    spatial_config["features"] = 1
    spatial_config["ppm_encoder_kwargs"] = {"ppm_encoding": False}
    spatial_encoder = networks.MultiresEncoding(None, **spatial_config)
    ppm_encoder = networks.FourierFeatures(1, freq_num=8, freq_scale=configs["FourierFeatures"]["freq_scale"])
    encoder = networks.SplitCoordinateEncoding(spatial_encoder, ppm_encoder, spatial_dims=2)
    encoder_meta = {
        "spatial_encoder": {
            "name": "HashEncoding",
            "nlevels": 16,
            "features": 1,
            "out_size": spatial_encoder.out_size,
            "table_size": spatial_config["table_size"],
            "minres": spatial_config["minres"],
            "maxres": spatial_config["maxres"],
        },
        "ppm_encoder": {
            "freq_num": 8,
            "freq_scale": configs["FourierFeatures"]["freq_scale"],
            "out_size": ppm_encoder.out_size,
        },
    }
    return encoder, encoder_meta


def _build_fourier_lorentz_encoder(args: NS, device: str):
    spatial_encoder = networks.FourierFeatures(2, freq_num=8, freq_scale=configs["FourierFeatures"]["freq_scale"])
    ppm_encoder_kwargs = deepcopy(configs["MultiresEncoding"]["ppm_encoder_kwargs"])
    ppm_encoder_kwargs.pop("ppm_encoding", None)
    ppm_encoder_kwargs["dim"] = 16
    ppm_encoder_kwargs["device"] = device
    ppm_encoder_kwargs["dtype"] = torch.float32
    ppm_encoder = networks.LorentzEncoding(**ppm_encoder_kwargs)
    encoder = networks.SplitCoordinateEncoding(spatial_encoder, ppm_encoder, spatial_dims=2)
    encoder_meta = {
        "spatial_encoder": {
            "freq_num": 8,
            "freq_scale": configs["FourierFeatures"]["freq_scale"],
            "out_size": spatial_encoder.out_size,
        },
        "ppm_encoder": ppm_encoder_kwargs,
    }
    return encoder, encoder_meta


def _to_json_safe(value):
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def setup_model(dataset_split, args: NS):
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    pos_encoder = None
    pos_encoder_args = None
    encoder_label = args.pos_encoder

    if args.pos_encoder in ["MultiresEncoding", "LSE"]:
        if args.pos_encoder == "LSE":
            args.ppm_encoding = True
        multires_config = _build_multires_config(args, device)
        pos_encoder = networks.MultiresEncoding
        pos_encoder_args = multires_config
        metadata_config = deepcopy(multires_config)
        if args.ppm_encoding:
            encoder_label = "LorentzEncoding"
            args.__dict__.update({"pos_encoder_kwargs": metadata_config})
        else:
            encoder_label = "HashEncoding"
            metadata_config.pop("ppm_encoder_kwargs")
            args.__dict__.update({"pos_encoder_kwargs": metadata_config})
    elif args.pos_encoder == "HashFourierEncoding":
        pos_encoder, encoder_meta = _build_hash_fourier_encoder(args, device)
        args.__dict__.update({"pos_encoder_kwargs": encoder_meta})
    elif args.pos_encoder == "FourierLorentzEncoding":
        pos_encoder, encoder_meta = _build_fourier_lorentz_encoder(args, device)
        args.__dict__.update({"pos_encoder_kwargs": encoder_meta})
    elif args.pos_encoder is not None:
        pos_encoder = getattr(networks, args.pos_encoder)
        pos_encoder_args = configs[args.pos_encoder]

    args.__dict__.update({"encoder_label": encoder_label})

    inr = networks.INR(
        in_size=dataset_split.train_dataset.coord_size,
        out_size=dataset_split.train_dataset.value_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        pos_encoder=pos_encoder,
        pos_encoder_args=pos_encoder_args,
        layer_class=getattr(activation_functions, args.activation_layer),
        **configs[args.activation_layer]["kargs"] if args.activation_layer in configs else {},
    ).to(device)

    if not str(args.activation_layer).startswith("ReLU"):
        getattr(activation_functions, configs[args.activation_layer]["func"])(
            inr, configs[args.activation_layer]["kargs"]["omega"]
        )
        args.__dict__.update(**configs[args.activation_layer])

    optimizer = getattr(torch.optim, args.optimizer)(inr.parameters(), lr=args.lr)
    args.__dict__.update(**lr_scheduler_params)
    scheduler_class = getattr(torch.optim.lr_scheduler, args.lr_scheduler)
    params = lr_scheduler_params.get(args.lr_scheduler, {})
    lr_scheduler = scheduler_class(optimizer, **params)
    loss_fn = WeightedMSELoss(weight_method=args.weight_method, scale=dataset_split.scale_factor)
    return inr, optimizer, lr_scheduler, loss_fn


def _build_exp_name(args: NS) -> str:
    encoder_label = getattr(args, "encoder_label", args.pos_encoder)
    if encoder_label == "FourierFeatures":
        return f"PE_{args.split_method}"
    if encoder_label == "HashEncoding":
        return f"HashEncoding_{args.split_method}"
    if args.pos_encoder == "LSE" or encoder_label == "LorentzEncoding":
        suffix = "_clip" if args.pos_encoder_kwargs["ppm_encoder_kwargs"]["clip"] else ""
        peak_num = len(args.pos_encoder_kwargs["ppm_encoder_kwargs"]["center_ranges"])
        prefix = "LSE" if args.pos_encoder == "LSE" else "LorentzEncoding"
        return (
            f"{prefix}_{args.split_method}_{args.pos_encoder_kwargs['nlevels']}_"
            f"SpatialDims_{args.pos_encoder_kwargs['ppm_encoder_kwargs']['dim']}_"
            f"LorentzDims_{peak_num}_Peaks{suffix}"
        )
    if encoder_label == "HashFourierEncoding":
        return (
            f"HashFourierEncoding_{args.split_method}_"
            f"XY{args.pos_encoder_kwargs['spatial_encoder']['out_size']}_"
            f"PPM{args.pos_encoder_kwargs['ppm_encoder']['out_size']}"
        )
    if encoder_label == "FourierLorentzEncoding":
        suffix = "_clip" if args.pos_encoder_kwargs["ppm_encoder"].get("clip", True) else ""
        peak_num = len(args.pos_encoder_kwargs["ppm_encoder"]["center_ranges"])
        return (
            f"FourierLorentzEncoding_{args.split_method}_"
            f"XY{args.pos_encoder_kwargs['spatial_encoder']['out_size']}_"
            f"PPM{args.pos_encoder_kwargs['ppm_encoder']['dim']}_"
            f"LorentzDims_{peak_num}_Peaks{suffix}"
        )
    if args.activation_layer in ["SIRENLayer", "WIRELayer"]:
        return f"{args.activation_layer}_{args.split_method}"
    raise ValueError(f"Unsupported encoder or activation: {encoder_label}, {args.activation_layer}")


def run(args, slice_index):
    dataset_split, train_dataloader, test_dataloader = setup_dataloader(args, slice_index)
    inr_model, optimizer, lr_scheduler, loss_fn = setup_model(dataset_split, args)

    current_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    exp_name = _build_exp_name(args)
    exp_path = str(Path(args.log_dir) / exp_name / current_time)

    os.makedirs(exp_path, exist_ok=True)
    log_dir, _ = create_directories(exp_path, slice_index)

    with open(os.path.join(exp_path, "args.json"), "w", encoding="utf-8") as handle:
        json.dump(_to_json_safe(vars(args)), handle, indent=4)

    trainer = Trainer(
        dataset_split,
        inr_model,
        optimizer,
        lr_scheduler,
        loss_fn,
        log_dir,
        device=f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu",
        patience=args.patience,
    )
    trainer(
        train_dataloader,
        test_dataloader,
        cal_metrics=True,
        num_epochs=args.num_epochs,
    )
    return trainer.pred_img, dataset_split.affine


def process_by_slice(args):
    slices = []
    slice_index = 0
    affine = None
    while True:
        try:
            pred_img, affine = run(args, slice_index)
            slices.append(pred_img)
            slice_index += 1
        except Exception as exc:
            print(f"Stopped processing at slice {slice_index}: {exc}")
            break

    if not slices:
        print("No slices were processed.")
        return

    volume = np.stack(slices, axis=2).transpose(0, 2, 1, 3)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(volume, affine=affine), str(output_path))


def process_by_case(args):
    slice_index = args.slice_index if args.slice else None
    run(args, slice_index)
