import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from INR_dataset import DatasetSplit
from .utils import plot_2d_or_3d_image


def calculate_psnr(pred, ref, mask=None):
    if mask is not None:
        pred *= mask
    mse = torch.mean((pred - ref) ** 2, dim=(-3, -2))
    max_value = ref.max()
    psnr_val = 20 * torch.log10(max_value / torch.sqrt(mse))
    return psnr_val.mean()


class BCTrainer:
    def __init__(
        self,
        dataset_spilt: DatasetSplit,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        loss_fn: nn.Module,
        log_dir: str,
        device: torch.device,
        patience: int = 500,
        is_slice: bool = False,
    ):
        self.train_gt_imgs = dataset_spilt.train_dataset.image
        self.test_gt_imgs = dataset_spilt.test_dataset.image
        self.train_mask = dataset_spilt.train_dataset.mask
        self.test_mask = dataset_spilt.test_dataset.mask
        self.train_indices = dataset_spilt.train_dataset.indices
        self.test_indices = dataset_spilt.test_dataset.indices
        self.affine = dataset_spilt.affine
        self.offsets = dataset_spilt.offsets
        self.scale_factor = dataset_spilt.scale_factor
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.name = str(Path(log_dir).parent.parent.name)
        self.epoch = 0
        self.eval_interval = None
        self.best_loss = np.inf
        self.best_psnr = -np.inf
        self.best_epoch = 0
        self.writer = SummaryWriter(log_dir)
        self.patience = patience
        self.early_stop_counter = 0
        self.is_slice = is_slice
        self.train_coords_norm = self._get_coords_norm(self.train_gt_imgs.shape[:-1], self.train_indices)
        self.test_coords_norm = self._get_coords_norm(self.test_gt_imgs.shape[:-1], self.test_indices)
        self.model_paths = []
        self.pred_img = None

    def _log_lorentz_params(self, epoch: int):
        ppm_encoder = None
        if hasattr(self.model, "pos_encoder") and hasattr(self.model.pos_encoder, "ppm_encoder"):
            ppm_encoder = self.model.pos_encoder.ppm_encoder
        if ppm_encoder is None or not hasattr(ppm_encoder, "gamma") or not hasattr(ppm_encoder, "mu"):
            return

        gamma = ppm_encoder.gamma.detach().cpu() * self.scale_factor
        mu = ppm_encoder.mu.detach().cpu() * self.scale_factor
        amplitude = ppm_encoder.amplitude.detach().cpu() if ppm_encoder.amplitude is not None else torch.zeros_like(mu)

        hot_map = torch.stack((mu, gamma, amplitude), dim=-1).unsqueeze(0).unsqueeze(0).transpose(2, 3)
        hot_map = (hot_map - hot_map.min()) / (hot_map.max() - hot_map.min() + 1e-8)
        self.writer.add_image(f"PPMEncoder/{self.name}", hot_map, epoch + 1, dataformats="NCHW")

        for channel in range(amplitude.shape[-1]):
            self.writer.add_scalar(f"PPMEncoder/mu_channel_{channel}/{self.name}", mu[..., channel].mean().item(), epoch + 1)
            self.writer.add_scalar(
                f"PPMEncoder/amplitude_channel_{channel}/{self.name}",
                amplitude[..., channel].mean().item(),
                epoch + 1,
            )
            self.writer.add_scalar(
                f"PPMEncoder/gamma_channel_{channel}/{self.name}",
                gamma[..., channel].mean().item(),
                epoch + 1,
            )
        self.writer.add_histogram(f"PPMEncoder/mu_hist/{self.name}", mu, epoch + 1)
        self.writer.add_histogram(f"PPMEncoder/amplitude_hist/{self.name}", amplitude, epoch + 1)
        self.writer.add_histogram(f"PPMEncoder/gamma_hist/{self.name}", gamma, epoch + 1)

    @torch.no_grad()
    def sample_at_resolution(self, resolution, coords_norm):
        points_num = max(1, np.prod(resolution) // 64)
        predictions = []
        for start in range(0, coords_norm.shape[0], points_num):
            batch_coords = coords_norm[start : start + points_num]
            predictions.append(self.model(batch_coords))
        predictions_flat = torch.cat(predictions, dim=0)
        if predictions_flat.ndim == 1 or predictions_flat.shape[-1] == 1:
            return predictions_flat.reshape(*resolution)
        return predictions_flat.reshape(*resolution, -1)

    def _get_coords_norm(self, resolution, indices):
        meshgrid = torch.meshgrid([torch.arange(0, dim, device=self.device) for dim in resolution], indexing="ij")
        coords = torch.stack(meshgrid, dim=-1).reshape(-1, len(resolution)).type(torch.float32)
        resolution_tensor = torch.tensor(resolution, device=self.device, dtype=torch.float32)
        coords_norm = (coords / resolution_tensor * 2 - 1).type(torch.float32)
        z_indices = coords[:, -1].long()
        offsets_tensor = torch.tensor(self.offsets, device=self.device, dtype=torch.float32)
        return torch.cat([coords_norm[:, :-1], offsets_tensor[indices[z_indices]].unsqueeze(1)], dim=1)

    def _save_zeros(self, resolution, path):
        cest_imgs_pred = np.flip(np.zeros(list(resolution) + [len(self.offsets)]), axis=1)
        self.pred_img = cest_imgs_pred
        with open(str(path), "wb") as handle:
            np.save(handle, cest_imgs_pred)

    def _save(self, train_imgs, test_imgs):
        pred = torch.zeros(list(train_imgs.shape[:-1]) + [len(self.offsets)])
        pred[..., self.train_indices] = train_imgs.detach().cpu()
        pred[..., self.test_indices] = test_imgs.detach().cpu()
        return pred.numpy()

    def _calculate_psnr(self, gt_imgs, coords_norm, mask):
        img_preds = self.sample_at_resolution(gt_imgs.shape[:-1], coords_norm.to(self.device))
        psnr = calculate_psnr(
            img_preds,
            gt_imgs.squeeze().to(img_preds.device),
            mask.squeeze().to(img_preds.device) if mask is not None else None,
        ).cpu().item()
        return img_preds, psnr

    def on_epoch_end(self, loss: float, epoch: int, eval_interval: int, cal_metrics: bool = True, *args):
        pred_file_path = Path(self.writer.get_logdir()).parent / f"{self.name}.nii.gz"
        if loss < self.best_loss and (self.epoch + 1 >= 10 * self.eval_interval) and cal_metrics:
            train_img_preds, train_psnr = self._calculate_psnr(
                self.train_gt_imgs, self.train_coords_norm, self.train_mask
            )
            test_img_preds, test_psnr = self._calculate_psnr(
                self.test_gt_imgs, self.test_coords_norm, self.test_mask
            )
            self.writer.add_scalar("PSNR/Train", train_psnr, epoch + 1)
            self.writer.add_scalar("PSNR/Test", test_psnr, epoch + 1)
            self.writer.flush()
            self._log_lorentz_params(epoch)
            self._add_images(train_img_preds, self.train_gt_imgs.squeeze(), "Train", *args)
            self._add_images(test_img_preds, self.test_gt_imgs.squeeze(), "Test", *args)
            self.best_loss = loss
            self.best_epoch = self.epoch + 1
            self.early_stop_counter = 0
            model_path = (
                f"{Path(self.writer.get_logdir()).parent}/models/"
                f"best_model_epoch{self.epoch + 1}_val_metric={round(self.best_psnr, 2)}.pth"
            )
            if test_psnr > 20.0 and test_psnr > self.best_psnr:
                self.best_psnr = test_psnr
                self.model_paths.append(model_path)
                if len(self.model_paths) > 3:
                    os.remove(self.model_paths.pop(0))
                torch.save(self.model.state_dict(), model_path)
                cest_imgs_pred = self._save(train_img_preds, test_img_preds)
                if len(train_img_preds.shape) == 3:
                    self.pred_img = cest_imgs_pred
                    nib.save(nib.Nifti1Image(cest_imgs_pred, self.affine), str(pred_file_path))
            elif not pred_file_path.exists() and len(train_img_preds.shape) == 3:
                try:
                    cest_imgs_pred = self._save(train_img_preds, test_img_preds)
                    self.pred_img = cest_imgs_pred
                    nib.save(nib.Nifti1Image(cest_imgs_pred, self.affine), str(pred_file_path))
                except Exception:
                    self._save_zeros(train_img_preds.shape[:-1], str(pred_file_path).replace(".nii.gz", ".npy"))
            return False

        self.early_stop_counter += 1
        if self.early_stop_counter >= self.patience + 10 * eval_interval:
            print(f"\nEarly stopping triggered at epoch {self.epoch + 1}")
            return True
        if self.epoch + 1 >= 10000:
            print(f"\nMaximum epochs reached: {self.epoch + 1}")
            return True
        return False

    def _add_images(self, pred_imgs: torch.Tensor, gt_imgs: torch.Tensor, phase: str, *args):
        plot_idx = 48 if phase == "Train" else int(np.random.choice([35, 61], 1, replace=False)[0])
        gt_imgs = gt_imgs.to(pred_imgs.device).unsqueeze(dim=0)
        pred_imgs = pred_imgs.unsqueeze(dim=0)
        channel_idx = list(self.train_indices).index(plot_idx) if phase == "Train" else list(self.test_indices).index(plot_idx)
        offset = round(self.offsets[plot_idx] * self.scale_factor, 2)

        plot_2d_or_3d_image(
            data=pred_imgs,
            step=self.epoch + 1,
            writer=self.writer,
            index=channel_idx,
            tag=f"{self.name}/{phase} Pred ({offset} ppm)",
        )
        plot_2d_or_3d_image(
            data=gt_imgs,
            step=self.epoch + 1,
            writer=self.writer,
            index=channel_idx,
            tag=f"{self.name}/{phase} GT ({offset} ppm)",
        )
        plot_2d_or_3d_image(
            data=10 * abs(pred_imgs - gt_imgs),
            step=self.epoch + 1,
            writer=self.writer,
            index=channel_idx,
            tag=f"{self.name}/{phase} 10 x Diff ({offset} ppm)",
        )

        for idx, extra in enumerate(args):
            plot_2d_or_3d_image(
                data=extra,
                step=self.epoch + 1,
                writer=self.writer,
                index=plot_idx,
                max_channels=3,
                tag=f"{self.name}/{phase}_Image_{idx}",
            )
