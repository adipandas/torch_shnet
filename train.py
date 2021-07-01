import os
import os.path as osp
import numpy as np
import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from models import PoseNet, HeatMapLossBatch
from utils.dataset import MPIIAnnotationHandler
from utils.dataset import MPIIDataset
from utils.read_config import yaml_to_object, Configurations
import utils.helpers as utils


def setup(rank, world_size, master_addr='127.0.0.1', master_port=12355):
    """
    Initialize the distributed environment.

    Args:
        rank (int): Rank of current process
        world_size (int): Total number of processes
        master_addr (str): IP address of master.
        master_port (int or str): Port number of master

    """
    os.environ['MASTER_ADDR'] = str(master_addr)
    os.environ['MASTER_PORT'] = str(int(master_port))
    dist.init_process_group("gloo", rank=rank, world_size=world_size)   # initialize the process group


def cleanup():
    """
    Clean the distributed compute pipeline.
    """
    dist.destroy_process_group()


class DataPartitioner:
    """
    Partitions a dataset into different chuncks.

    Args:
        training_data_indices (numpy.ndarray):
        validation_data_indices (numpy.ndarray):
        sizes (tuple or list): Fraction of data to allote to a single partition
        seed (int): Seed for random number generator.

    """
    def __init__(self, training_data_indices, validation_data_indices, sizes=(0.7, 0.2, 0.1), seed=1234):
        np.random.seed(seed)
        self.training_data_indices = training_data_indices
        self.validation_data_indices = validation_data_indices
        np.random.shuffle(self.training_data_indices)       # shuffle indices in-place
        np.random.shuffle(self.validation_data_indices)     # shuffle indices in-place
        self.training_partitions = utils.create_partitions(self.training_data_indices, sizes)
        self.validation_partitions = utils.create_partitions(self.validation_data_indices, sizes)

    def use(self, partition_id, config, mpii_annotation_handle):
        """
        Get a data partition to use.

        Args:
            partition_id (int): Index of the data partition to use.
            config (Configurations): Configuration file object.
            mpii_annotation_handle (MPIIAnnotationHandler): MPII annotation data handler object

        Returns:
            tuple[MPIIDataset, MPIIDataset]: Tuple of length ``2`` containing:
                - training_data_partition (MPIIDataset): Partition of training data.
                - validation_data_partition (MPIIDataset): Partition of validation data.
        """

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

        training_data_partition = MPIIDataset(
            indices=self.training_partitions[partition_id],
            mpii_annotation_handle=mpii_annotation_handle,
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

        validation_data_partition = MPIIDataset(
            indices=self.validation_partitions[partition_id],
            mpii_annotation_handle=mpii_annotation_handle,
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
        return training_data_partition, validation_data_partition


def partition_dataset(rank, world_size, config):
    """
    Partitioning Dataset

    Args:
        rank (int): Rank of the current process
        world_size (int): Total number of porcesses.
        config (Configurations): Configurations.

    Returns:
        tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, int]: Tuple containing following elemenets in given order:
            - training_set (torch.utils.data.DataLoader):
            - validation_set (torch.utils.data.DataLoader):
            - batch_size (int):
    """
    seed = int(config.neural_network.train.random_seed)
    data_path = osp.join(config.root_dir, config.data.MPII.path.base)
    training_annotation_file = osp.join(data_path, config.data.MPII.path.annotations.training)
    validation_annotation_file = osp.join(data_path, config.data.MPII.path.annotations.validation)
    image_dir = osp.join(data_path, config.data.MPII.path.images)
    mpii_annotation_handle = MPIIAnnotationHandler(training_annotation_file, validation_annotation_file, image_dir)
    training_data_indices, validation_data_indices = mpii_annotation_handle.split_data()
    batch_size = int(config.neural_network.train.batch_size / float(world_size))
    partition_sizes = [1.0 / world_size for _ in range(world_size)]
    partition = DataPartitioner(training_data_indices, validation_data_indices, sizes=partition_sizes, seed=seed)
    training_data_partition, validation_data_partition = partition.use(partition_id=rank, config=config, mpii_annotation_handle=mpii_annotation_handle)
    training_sampler = torch.utils.data.distributed.DistributedSampler(training_data_partition, num_replicas=world_size, rank=rank)
    validation_sampler = torch.utils.data.distributed.DistributedSampler(validation_data_partition, num_replicas=world_size, rank=rank)
    training_dataloader = torch.utils.data.DataLoader(training_data_partition, batch_size=batch_size, shuffle=False, sampler=training_sampler, num_workers=int(config.neural_network.train.num_workers*0.5))
    validation_dataloader = torch.utils.data.DataLoader(validation_data_partition, batch_size=batch_size, shuffle=False, sampler=validation_sampler, num_workers=int(config.neural_network.train.num_workers*0.5))
    return training_dataloader, validation_dataloader, batch_size


def average_gradients(model, world_size):
    """
    Gradient averaging.

    Args:
        model (DistributedDataParallel):
        world_size (int): World size or total number of processes.

    Returns:

    """
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
        param.grad.data /= float(world_size)


def run(rank, world_size, config):
    """
    Distributed Synchronous SGD Example

    Args:
        rank (int): Rank of the current process.
        world_size (int): Total number of processes (a.k.a world size) in use.
        config (ConfigYamlParserMPII):

    Returns:

    """
    setup(rank, world_size, master_addr=config.neural_network.train.DistributedDataParallel.MASTER_ADDR, master_port=config.neural_network.train.DistributedDataParallel.MASTER_PORT)

    torch.manual_seed(int(config.neural_network.train.random_seed))
    training_dataloader, validation_dataloader, batch_size = partition_dataset(rank, world_size, config)

    total_epochs = int(config.neural_network.train.epochs)
    learning_rate = float(config.neural_network.train.learning_rate)

    n_hourglass = int(config.neural_network.PoseNet.n_hourglass)
    in_channels = int(config.neural_network.PoseNet.in_channels)
    out_channels = int(config.neural_network.PoseNet.out_channels)
    channel_increase = int(config.neural_network.PoseNet.channel_increase)
    model = PoseNet(n_hourglass=n_hourglass, in_channels=in_channels, out_channels=out_channels, channel_increase=channel_increase).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    loss_fn = HeatMapLossBatch()

    train_loader = iter(training_dataloader)
    valid_loader = iter(validation_dataloader)

    for epoch in range(total_epochs):
        training_dataloader.sampler.set_epoch(epoch)
        epoch_loss = 0.0

        train_iters = 0
        while train_iters < int(config.neural_network.train.train_iterations):
            train_iters += 1

            try:
                images, heatmaps = next(train_loader)
            except StopIteration:
                train_loader = iter(training_dataloader)
                images, heatmaps = next(train_loader)

            images = images.cuda(non_blocking=True)
            heatmaps = heatmaps.cuda(non_blocking=True)
            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output, heatmaps)
            epoch_loss += utils.make_output(loss)
            loss.backward()
            average_gradients(model, world_size)
            optimizer.step()

        # validation
        with torch.no_grad():
            validation_loss = 0
            validation_dataloader.sampler.set_epoch(epoch)

            valid_iters = 0
            while valid_iters < int(config.neural_network.train.valid_iterations):
                valid_iters += 1

                try:
                    images, heatmaps = next(valid_loader)
                except StopIteration:
                    train_loader = iter(validation_dataloader)
                    images, heatmaps = next(valid_loader)

                output = model(images)
                loss = loss_fn(output, heatmaps)
                validation_loss += utils.make_output(loss)

        epoch_train_loss = epoch_loss/config.neural_network.train.train_iterations
        epoch_valid_loss = validation_loss/config.neural_network.train.valid_iterations
        print(f"rank:{dist.get_rank():2d}  epoch:{epoch:3d}  epoch_train_loss:{epoch_train_loss:0.4f}  epoch_valid_loss:{epoch_valid_loss:0.4f}")

        save_checkpoint = (rank == 0 and epoch > 0 and config.neural_network.train.checkpoint.save and epoch % config.neural_network.train.checkpoint.save_every == 0)
        if save_checkpoint:
            torch.save(model.state_dict(), config.neural_network.train.checkpoint.path)  # saving it in one process is sufficient.
        dist.barrier()

    cleanup()


def load_configuration(configuration_file_path="./config.yaml"):
    configuration = yaml_to_object(configuration_file_path)
    setattr(configuration, "root_dir", osp.dirname(osp.abspath(__file__)))
    return configuration


def runner(fn, world_size, config):
    mp.spawn(fn, args=(world_size, config), nprocs=world_size, join=True)


if __name__ == "__main__":
    config = load_configuration(configuration_file_path="./config.yaml")
    word_size = config.neural_networ.train.num_workers
    runner(run, word_size, config)
