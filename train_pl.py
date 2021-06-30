import os.path as osp

import torch
from torch.utils.data.dataloader import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from utils.read_config import yaml_to_object
from utils.dataset import MPIIDataset
from utils.annotation_handler import MPIIAnnotationHandler
from models import PoseNet, HeatMapLossBatch


torch.cuda.empty_cache()


def data_loader_creator(config):
    data_path = osp.join(config.root_dir, config.data.MPII.path.base)
    training_annotation_file = osp.join(data_path, config.data.MPII.path.annotations.training)
    validation_annotation_file = osp.join(data_path, config.data.MPII.path.annotations.validation)
    image_dir = osp.join(data_path, config.data.MPII.path.images)

    data_handle = MPIIAnnotationHandler(training_annotation_file, validation_annotation_file, image_dir)
    train_indices, valid_indices = data_handle.split_data()

    image_scale_factor_range = (float(config.neural_network.train.data_augmentation.image_scale_factor.min), float(config.neural_network.train.data_augmentation.image_scale_factor.max))
    input_resolution = int(config.neural_network.train.input_resolution)
    output_resolution = int(config.neural_network.train.output_resolution)
    num_parts = int(config.data.MPII.parts.max_count)
    reference_image_size = int(config.data.MPII.reference_image_size)
    max_rotation_angle = float(config.neural_network.train.data_augmentation.rotation_angle_max)
    image_color_jitter_probability = float(config.neural_network.train.data_augmentation.image_color_jitter_probability)
    image_horizontal_flip_probability = float(config.neural_network.train.data_augmentation.image_horizontal_flip_probability)
    hue_max_delta = float(config.neural_network.train.data_augmentation.hue_max_delta)
    saturation_min_delta = float(config.neural_network.train.data_augmentation.saturation_min_delta)
    brightness_max_delta = float(config.neural_network.train.data_augmentation.brightness_max_delta)
    contrast_min_delta = float(config.neural_network.train.data_augmentation.contrast_min_delta)

    train_data = MPIIDataset(
        indices=train_indices, mpii_annotation_handle=data_handle,
        horizontally_flipped_keypoint_ids=config.data.MPII.parts.flipped_ids,
        input_resolution=input_resolution,
        output_resolution=output_resolution,
        num_parts=num_parts,
        reference_image_size=reference_image_size,
        max_rotation_angle=max_rotation_angle,
        image_scale_factor_range=image_scale_factor_range,
        image_color_jitter_probability=image_color_jitter_probability,
        image_horizontal_flip_probability=image_horizontal_flip_probability,
        hue_max_delta=hue_max_delta,
        saturation_min_delta=saturation_min_delta,
        brightness_max_delta=brightness_max_delta,
        contrast_min_delta=contrast_min_delta
    )

    valid_data = MPIIDataset(
        indices=valid_indices, mpii_annotation_handle=data_handle,
        horizontally_flipped_keypoint_ids=config.data.MPII.parts.flipped_ids,
        input_resolution=input_resolution,
        output_resolution=output_resolution,
        num_parts=num_parts,
        reference_image_size=reference_image_size,
        max_rotation_angle=max_rotation_angle,
        image_scale_factor_range=image_scale_factor_range,
        image_color_jitter_probability=image_color_jitter_probability,
        image_horizontal_flip_probability=image_horizontal_flip_probability,
        hue_max_delta=hue_max_delta,
        saturation_min_delta=saturation_min_delta,
        brightness_max_delta=brightness_max_delta,
        contrast_min_delta=contrast_min_delta
    )

    train_dataloader = DataLoader(train_data, batch_size=config.neural_network.train.batch_size, num_workers=4)
    valid_dataloader = DataLoader(valid_data, batch_size=config.neural_network.train.batch_size, num_workers=4)

    return train_dataloader, valid_dataloader


class PoseNetLightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.posenet = PoseNet(config.neural_network.PoseNet.n_hourglass,
                               config.neural_network.PoseNet.in_channels,
                               config.neural_network.PoseNet.out_channels,
                               config.neural_network.PoseNet.channel_increase)
        self.heatmap_loss_batch = HeatMapLossBatch()

    def forward(self, x):
        out = self.posenet(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.posenet(x)
        loss = self.heatmap_loss_batch(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.posenet(x)
        loss = self.heatmap_loss_batch(y_hat, y)
        self.log('valid_loss', loss)
        return loss

    def configure_optimizers(self):
        lr = self.config.neural_network.train.learning_rate
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer


def load_configuration(configuration_file_path="./config.yaml"):
    configuration = yaml_to_object(configuration_file_path)
    setattr(configuration, "root_dir", osp.dirname(osp.abspath(__file__)))
    return configuration


config = load_configuration(configuration_file_path="./config.yaml")
train_dataloader, valid_dataloader = data_loader_creator(config)

min_epochs = config.neural_network.train.epochs         # number of cycles over dataset

# default logger used by trainer
logger = TensorBoardLogger(save_dir=osp.join(config.root_dir, config.neural_network.train.logs.path), version=1, name='posenet_logs')
posenet = PoseNetLightning(config)
trainer = pl.Trainer(gpus=1, min_epochs=min_epochs, logger=logger)
trainer.fit(posenet, train_dataloader, valid_dataloader)
