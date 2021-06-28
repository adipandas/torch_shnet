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
from utils.data import MPIIAnnotationHandler
from utils.data import MPIIDataset
from utils.read_config import yaml_to_object, Configurations


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

        self.training_partitions = self._create_partition_sets(self.training_data_indices, sizes)
        self.validation_partitions = self._create_partition_sets(self.validation_data_indices, sizes)

    def _create_partition_sets(self, indices, sizes):
        """
        Create partition sets with defined sizes.

        Args:
            indices (numpy.ndarray): Data indices for using to create data partitions.
            sizes (tuple): Tuple containing fraction size of each partition.

        Returns:
            list[numpy.ndarray]: List with each element as numpy array of data indices to be used in each partition. Length of list is same as length of ``sizes`` container given as the input to this method.
        """
        partitions = []

        data_len = indices.shape[0]

        for fraction in sizes:
            part_len = int(fraction * data_len)
            partitions.append(indices[0:part_len])
            indices = indices[part_len:]
        return partitions

    def use(self, partition_id, config, mpii_annotation_handle):
        """
        Get a data partition to use.

        Args:
            partition_id (int): Index of the data partition to use.
            config (Configurations): Configuration file object.
            mpii_annotation_handle (MPIIAnnotationHandler): MPII annotation data handler object

        Returns:
            tuple[MPIIDataset, MPIIDataset]: Tuple of length ``2`` containing:
                - training_data_partition (Partition):
                - validation_data_partition (Partition):
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
    # world_size = min(config.NN_TRAINING_PARAMS['DistributedDataParallel']['world_size'], dist.get_world_size())

    batch_size_total = int(config.neural_network.train.batch_size)

    data_path = osp.join(config.root_dir, config.data.MPII.path.base)
    training_annotation_file = osp.join(data_path, config.data.MPII.path.annotations.training)
    validation_annotation_file = osp.join(data_path, config.data.MPII.path.annotations.validation)
    image_dir = osp.join(data_path, config.data.MPII.path.images)

    mpii_annotation_handle = MPIIAnnotationHandler(
        training_annotation_file=training_annotation_file,
        validation_annotation_file=validation_annotation_file,
        image_dir=image_dir,
        horizontally_flipped_keypoint_ids=config.data.MPII.parts.flipped_ids
    )

    training_data_indices, validation_data_indices = mpii_annotation_handle.split_data()

    batch_size = int(batch_size_total / float(world_size))
    partition_sizes = [1.0 / world_size for _ in range(world_size)]

    partition = DataPartitioner(training_data_indices, validation_data_indices, sizes=partition_sizes, seed=seed)

    # rank = dist.get_rank()
    training_data_partition, validation_data_partition = partition.use(partition_id=rank, config=config, mpii_annotation_handle=mpii_annotation_handle)

    training_sampler = torch.utils.data.distributed.DistributedSampler(training_data_partition, num_replicas=world_size, rank=rank)
    validation_sampler = torch.utils.data.distributed.DistributedSampler(validation_data_partition, num_replicas=world_size, rank=rank)

    training_dataloader = torch.utils.data.DataLoader(training_data_partition, batch_size=batch_size, shuffle=False, sampler=training_sampler)
    validation_dataloader = torch.utils.data.DataLoader(validation_data_partition, batch_size=batch_size, shuffle=False, sampler=validation_sampler)

    return training_dataloader, validation_dataloader, batch_size


def average_gradients(model, world_size):
    """
    Gradient averaging.

    Args:
        model (DistributedDataParallel):
        world_size (int): World size or total number of processes.

    Returns:

    """
    size = float(world_size)
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
        param.grad.data /= size


def run(rank, world_size, config):
    """
    Distributed Synchronous SGD Example

    Args:
        rank (int): Rank of the current process.
        world_size (int): Total number of processes (a.k.a world size) in use.
        config (ConfigYamlParserMPII):

    Returns:

    """
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

    total_step = len(training_dataloader)
    for epoch in range(total_epochs):
        training_dataloader.sampler.set_epoch(epoch)
        epoch_loss = 0.0
        for i, (images, heatmaps) in enumerate(training_dataloader):
            images = images.cuda(non_blocking=True)
            heatmaps = heatmaps.cuda(non_blocking=True)
            # data, target = Variable(data), Variable(target)
            # data, target = Variable(data.cuda(rank)), Variable(target.cuda(rank))
            output = model(images)
            optimizer.zero_grad()
            loss = loss_fn(output, heatmaps)
            epoch_loss += loss.data[0]
            loss.backward()
            average_gradients(model, world_size)
            optimizer.step()
        print('Rank ', dist.get_rank(), ', epoch ', epoch, ': ', epoch_loss / num_batches)

        if rank == 0 and epoch > 0 and config.neural_network.train.checkpoint.save and epoch % config.neural_network.train.checkpoint.save_every == 0:
            torch.save(model.state_dict(), config.neural_network.train.checkpoint.path)  # saving it in one process is sufficient.

    dist.barrier()

    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn, args=(world_size,), nprocs=world_size, join=True)


def load_configuration(configuration_file_path="./config.yaml"):
    configuration = yaml_to_object(configuration_file_path)
    setattr(configuration, "root_dir", osp.dirname(osp.abspath(__file__)))
    return configuration


if __name__ == "__main__":
    config = load_configuration(configuration_file_path="./config.yaml")
