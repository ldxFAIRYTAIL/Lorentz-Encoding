from typing import List, Optional, Union

import numpy as np
import torch
from monai.config import NdarrayTensor
from monai.transforms import rescale_array
from monai.utils import convert_data_type, optional_import
from tensorboard.compat.proto.summary_pb2 import Summary
from torch.utils.tensorboard import SummaryWriter

PIL, _ = optional_import("PIL")
GifImage, _ = optional_import("PIL.GifImagePlugin", name="Image")
SummaryX, _ = optional_import("tensorboardX.proto.summary_pb2", name="Summary")
SummaryWriterX, has_tensorboardx = optional_import("tensorboardX", name="SummaryWriter")


def _image3_animated_gif(
    tag: str,
    image: Union[np.ndarray, torch.Tensor],
    writer,
    frame_dim: int = 0,
    scale_factor: float = 1.0,
):
    if len(image.shape) != 3:
        raise AssertionError("3D image tensors expected to be in `HWD` format.")

    image_np, *_ = convert_data_type(image, output_type=np.ndarray)
    ims = [(i * scale_factor).astype(np.uint8, copy=False) for i in np.moveaxis(image_np, frame_dim, 0)]
    ims = [GifImage.fromarray(im) for im in ims]
    img_str = b""
    for b_data in PIL.GifImagePlugin.getheader(ims[0])[0]:
        img_str += b_data
    img_str += b"\x21\xFF\x0B\x4E\x45\x54\x53\x43\x41\x50" b"\x45\x32\x2E\x30\x03\x01\x00\x00\x00"
    for image_frame in ims:
        for b_data in PIL.GifImagePlugin.getdata(image_frame):
            img_str += b_data
    img_str += b"\x3B"

    summary = SummaryX if has_tensorboardx and isinstance(writer, SummaryWriterX) else Summary
    summary_image_str = summary.Image(height=10, width=10, colorspace=1, encoded_image_string=img_str)
    image_summary = summary.Value(tag=tag, image=summary_image_str)
    return summary(value=[image_summary])


def make_animated_gif_summary(
    tag: str,
    image: Union[np.ndarray, torch.Tensor],
    writer=None,
    max_out: int = 3,
    frame_dim: int = -3,
    scale_factor: float = 1.0,
) -> Summary:
    suffix = "/image" if max_out == 1 else "/image/{}"
    frame_dim = frame_dim - 1 if frame_dim > 0 else frame_dim

    summary_op = []
    for it_i in range(min(max_out, list(image.shape)[0])):
        one_channel_img = (
            image[it_i, :, :, :].squeeze(dim=0) if isinstance(image, torch.Tensor) else image[it_i, :, :, :]
        )
        summary_op.append(
            _image3_animated_gif(tag + suffix.format(it_i), one_channel_img, writer, frame_dim, scale_factor)
        )
    return summary_op


def add_animated_gif(
    writer: SummaryWriter,
    tag: str,
    image_tensor: Union[np.ndarray, torch.Tensor],
    max_out: int = 3,
    frame_dim: int = -3,
    scale_factor: float = 1.0,
    global_step: Optional[int] = None,
) -> None:
    summary = make_animated_gif_summary(
        tag=tag,
        image=image_tensor,
        writer=writer,
        max_out=max_out,
        frame_dim=frame_dim,
        scale_factor=scale_factor,
    )
    for item in summary:
        writer._get_file_writer().add_summary(item, global_step)


def plot_2d_or_3d_image(
    data: Union[NdarrayTensor, List[NdarrayTensor]],
    step: int,
    writer: SummaryWriter,
    index: int = 0,
    max_channels: int = 1,
    frame_dim: int = -1,
    max_frames: int = 100,
    tag: str = "output",
) -> None:
    data_index = data[..., index]
    frame_dim = frame_dim - 1 if frame_dim > 0 else frame_dim

    array = data_index.detach().cpu().numpy() if isinstance(data_index, torch.Tensor) else data_index
    array = np.flip(np.swapaxes(array, -1, -2), axis=-2)

    if array.ndim == 2:
        array = rescale_array(array, 0, 1)
        writer.add_image(tag, array, step, dataformats="HW")
        return

    if array.ndim == 3:
        if array.shape[0] == 3 and max_channels == 3:
            writer.add_image(tag, array, step, dataformats="CHW")
            return
        for channel in array[:max_channels]:
            writer.add_image(tag, rescale_array(channel, 0, 1), step, dataformats="HW")
        return

    if array.ndim >= 4:
        spatial = array.shape[-3:]
        array = array.reshape([-1] + list(spatial))
        if array.shape[0] == 3 and max_channels == 3 and has_tensorboardx and isinstance(writer, SummaryWriterX):
            array = np.moveaxis(array, frame_dim, -1)
            writer.add_video(tag, array[None], step, fps=max_frames, dataformats="NCHWT")
            return
        max_channels = min(max_channels, array.shape[0])
        array = np.stack([rescale_array(channel, 0, 255) for channel in array[:max_channels]], axis=0)
        add_animated_gif(writer, tag, array, max_out=max_channels, frame_dim=frame_dim, global_step=step)
