import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from INR_dataset import DatasetSplit
from .BasicTrainer import BCTrainer


class Trainer(BCTrainer):
    def __init__(
        self,
        dataset_spilt: DatasetSplit,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        loss_fn: nn.Module,
        log_dir: str,
        device: str,
        patience: int = 500,
        is_slice: bool = False,
    ):
        super().__init__(
            dataset_spilt,
            model,
            optimizer,
            lr_scheduler,
            loss_fn,
            log_dir,
            device,
            patience,
            is_slice,
        )

    def _process_batch_data(self, *batch_data_tuple):
        moved_data = []
        for item in batch_data_tuple:
            if isinstance(item, torch.Tensor) and item is not None:
                moved_data.append(item.to(self.device))
            elif item is None:
                moved_data.append(None)
            else:
                moved_data.append(item)

        if len(moved_data) != 4:
            raise ValueError(f"Expected 4 items from DataLoader, got {len(moved_data)}.")

        indices, coords, values, mask = moved_data
        if indices is not None:
            indices = indices.view(-1, indices.shape[-1])
        if coords is not None:
            coords = coords.view(-1, coords.shape[-1])
        if values is not None:
            values = values.view(-1, 1) if values.ndim <= 2 else values.view(-1, values.shape[-1])
        if mask is not None:
            mask = mask.view(-1, 1) if mask.ndim <= 2 else mask.view(-1, mask.shape[-1])
            if mask.dtype == torch.bool:
                mask = mask.float()
        return indices, coords, values, mask

    def _train(self, *batch_data_tuple):
        _, coords, values, mask = self._process_batch_data(*batch_data_tuple)
        self.optimizer.zero_grad()
        preds = self.model(coords)
        preds = preds * mask if mask is not None else preds
        loss = self.loss_fn(coords, preds, values)
        loss_scalar = loss.mean() if loss.ndim > 0 else loss
        loss_scalar.backward()
        self.optimizer.step()
        if hasattr(self.model, "pos_encoder") and hasattr(self.model.pos_encoder, "ppm_encoding"):
            if self.model.pos_encoder.ppm_encoding:
                self.model.pos_encoder.ppm_encoder.clip_params()
        return loss

    def _test(self, *batch_data_tuple):
        _, coords, values, mask = self._process_batch_data(*batch_data_tuple)
        preds = self.model(coords)
        preds = preds * mask if mask is not None else preds
        loss = self.loss_fn(coords, preds, values)
        return loss.mean() if loss.ndim > 0 else loss

    def __call__(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        num_epochs: int = 5000,
        cal_metrics: bool = True,
        eval_interval: int = 100,
    ):
        self.eval_interval = eval_interval
        if hasattr(self, "train_gt_imgs") and self.train_gt_imgs is not None and torch.all(self.train_gt_imgs == 0):
            shape_to_save = self.train_gt_imgs.shape[:-1]
            self._save_zeros(shape_to_save)
            print("All zeros in the training set, skip training.")
            return

        with tqdm(total=num_epochs, desc="Training", ncols=80) as pbar:
            for epoch in range(num_epochs):
                self.epoch = epoch
                self.model.train()
                for batch_idx, batch_data_tuple in enumerate(train_loader):
                    train_loss = self._train(*batch_data_tuple)
                    if train_loss is not None:
                        train_loss_ = train_loss.mean() if train_loss.ndim > 0 else train_loss.clone()
                        self.writer.add_scalar(
                            "LearningRate",
                            self.lr_scheduler.get_last_lr()[0],
                            epoch * len(train_loader) + batch_idx,
                        )
                        self.writer.add_scalar(
                            "Loss/Train",
                            train_loss_.item(),
                            epoch * len(train_loader) + batch_idx,
                        )

                self.writer.add_scalar("Loss/Train", train_loss_, epoch)
                pbar.set_postfix({"Loss": "{:.2e}".format(train_loss_)})
                pbar.update(1)

                self.model.eval()
                with torch.no_grad():
                    for batch_data_tuple in test_loader:
                        test_loss = self._test(*batch_data_tuple)
                        if test_loss is not None:
                            self.writer.add_scalar("Loss/Test", test_loss, epoch)

                if self.on_epoch_end(train_loss_, epoch, eval_interval, cal_metrics):
                    print(f"Early stopping triggered at epoch {epoch + 1}.")
                    break
                self.lr_scheduler.step()

        self.writer.close()
        print(f"Training completed for {self.name}.")
