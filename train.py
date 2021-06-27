import os
from math import ceil
import numpy as np
import tempfile

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from models import PoseNet, HeatMapLossBatch
from utils.data import MPIIAnnotationHandler
from utils.data import MPIIDataset
from utils.read_config import ConfigYamlParserMPII


def setup(rank, world_size, master_addr='127.0.0.1', master_port='12355'):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group("gloo", rank=rank, world_size=world_size)   # initialize the process group


def cleanup():
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
            config (ConfigYamlParserMPII): Configuration file object.
            mpii_annotation_handle (MPIIAnnotationHandler): MPII annotation data handler object

        Returns:
            tuple[MPIIDataset, MPIIDataset]: Tuple of length ``2`` containing:
                - training_data_partition (Partition):
                - validation_data_partition (Partition):
        """

        _data_aug = config.NN_TRAINING_PARAMS['data_augmentation']
        image_scale_factor_range = (_data_aug['image_scale_factor']['min'], _data_aug['image_scale_factor']['max'])

        training_data_partition = MPIIDataset(
            indices=self.training_partitions[partition_id],
            mpii_annotation_handle=mpii_annotation_handle,
            input_resolution=config.NN_TRAINING_PARAMS['input_resolution'],
            output_resolution=config.NN_TRAINING_PARAMS['output_resolution'],
            num_parts=config.PARTS['max_count'],
            reference_image_size=config.REFERENCE_IMAGE_SIZE,
            max_rotation_angle=_data_aug['rotation_angle_max'],
            image_scale_factor_range=image_scale_factor_range,
            image_color_jitter_probability=_data_aug['image_color_jitter_probability'],
            image_horizontal_flip_probability=_data_aug['image_horizontal_flip_probability'],
            hue_max_delta=_data_aug['hue_max_delta'],
            saturation_min_delta=_data_aug['saturation_min_delta'],
            brightness_max_delta=_data_aug['brightness_max_delta'],
            contrast_min_delta=_data_aug['contrast_min_delta']
        )
        validation_data_partition = MPIIDataset(
            indices=self.validation_partitions[partition_id],
            mpii_annotation_handle=mpii_annotation_handle,
            input_resolution=config.NN_TRAINING_PARAMS['input_resolution'],
            output_resolution=config.NN_TRAINING_PARAMS['output_resolution'],
            num_parts=config.PARTS['max_count'],
            reference_image_size=config.REFERENCE_IMAGE_SIZE,
            max_rotation_angle=_data_aug['rotation_angle_max'],
            image_scale_factor_range=image_scale_factor_range,
            image_color_jitter_probability=_data_aug['image_color_jitter_probability'],
            image_horizontal_flip_probability=_data_aug['image_horizontal_flip_probability'],
            hue_max_delta=_data_aug['hue_max_delta'],
            saturation_min_delta=_data_aug['saturation_min_delta'],
            brightness_max_delta=_data_aug['brightness_max_delta'],
            contrast_min_delta=_data_aug['contrast_min_delta']
        )
        return training_data_partition, validation_data_partition


def partition_dataset(config):
    """
    Partitioning Dataset

    Args:
        config (ConfigYamlParserMPII):

    Returns:
        tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, int]: Tuple containing following elemenets in given order:
            - training_set (torch.utils.data.DataLoader):
            - validation_set (torch.utils.data.DataLoader):
            - batch_size (int):
    """

    seed = config.NN_TRAINING_PARAMS['random_seed']

    world_size = min(config.NN_TRAINING_PARAMS['DistributedDataParallel']['world_size'], dist.get_world_size())

    batch_size_total = config.NN_TRAINING_PARAMS['batch_size']

    mpii_annotation_handle = MPIIAnnotationHandler(
        training_annotation_file=config.TRAINING_ANNOTATION_FILE,
        validation_annotation_file=config.VALIDATION_ANNOTATION_FILE,
        image_dir=config.IMAGE_DIR,
        keypoint_info=config.PARTS
    )
    training_data_indices, validation_data_indices = mpii_annotation_handle.split_data()

    batch_size = batch_size_total / float(world_size)
    partition_sizes = [1.0 / world_size for _ in range(world_size)]

    partition = DataPartitioner(training_data_indices, validation_data_indices, sizes=partition_sizes, seed=seed)

    rank = dist.get_rank()
    training_data_partition, validation_data_partition = partition.use(partition_id=rank, config=config, mpii_annotation_handle=mpii_annotation_handle)
    training_set = torch.utils.data.DataLoader(training_data_partition, batch_size=batch_size, shuffle=True)
    validation_set = torch.utils.data.DataLoader(validation_data_partition, batch_size=batch_size, shuffle=True)
    return training_set, validation_set, batch_size


def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
        param.grad.data /= size


# def run(rank, size, config):
#     """ Distributed Synchronous SGD Example """
#     torch.manual_seed(1234)
#     train_set, bsz = partition_dataset(config,)
#     model = model = PoseNet(
#         n_hourglass=config.POSENET_INPUT_PARAMS['n_hourglass'],
#         in_channels=config.POSENET_INPUT_PARAMS['in_channels'],
#         out_channels=config.POSENET_INPUT_PARAMS['out_channels'],
#         channel_increase=config.POSENET_INPUT_PARAMS['channel_increase']
#     ).to(rank)
#     model = model
#     # model = model.cuda(rank)
#     # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
#
#     optimizer = optim.Adam(
#         filter(lambda p: p.requires_grad, ddp_model.parameters()),
#         lr=config.NN_TRAINING_PARAMS['learning_rate']
#     )
#
#     num_batches = ceil(len(train_set.dataset) / float(bsz))
#     for epoch in range(10):
#         epoch_loss = 0.0
#         for data, target in train_set:
#             data, target = Variable(data), Variable(target)
# #            data, target = Variable(data.cuda(rank)), Variable(target.cuda(rank))
#             optimizer.zero_grad()
#             output = model(data)
#             loss = F.nll_loss(output, target)
#             epoch_loss += loss.data[0]
#             loss.backward()
#             average_gradients(model)
#             optimizer.step()
#         print('Rank ',
#               dist.get_rank(), ', epoch ', epoch, ': ',
#               epoch_loss / num_batches)


# def demo_basic(rank, world_size, config):
#     """
#
#     Args:
#         rank:
#         world_size ():
#         config (ConfigYamlParserMPII):
#
#     Returns:
#
#     """
#
#     print(f"Running basic DDP example on rank {rank}.")
#     setup(rank, world_size)
#
#     # create model and move it to GPU with id rank
#     model = PoseNet(
#         n_hourglass=config.POSENET_INPUT_PARAMS['n_hourglass'],
#         in_channels=config.POSENET_INPUT_PARAMS['in_channels'],
#         out_channels=config.POSENET_INPUT_PARAMS['out_channels'],
#         channel_increase=config.POSENET_INPUT_PARAMS['channel_increase']
#     ).to(rank)
#
#     ddp_model = DDP(model, device_ids=[rank])
#
#     loss_fn = HeatMapLossBatch()
#
#     optimizer = optim.Adam(
#         filter(lambda p: p.requires_grad, ddp_model.parameters()),
#         lr=config.NN_TRAINING_PARAMS['learning_rate']
#     )
#
#     optimizer.zero_grad()
#     outputs = ddp_model(torch.randn(20, 10))
#     labels = torch.randn(20, 5).to(rank)
#     loss_fn(outputs, labels).backward()
#     optimizer.step()
#
#     cleanup()


def demo_checkpoint(rank, world_size, config):
    """

    Args:
        rank:
        world_size ():
        config (ConfigYamlParserMPII):

    Returns:

    """
    print(f"Running DDP checkpoint example on rank {rank}.")
    setup(rank, world_size)

    model = PoseNet(
        n_hourglass=config.POSENET_INPUT_PARAMS['n_hourglass'],
        in_channels=config.POSENET_INPUT_PARAMS['in_channels'],
        out_channels=config.POSENET_INPUT_PARAMS['out_channels'],
        channel_increase=config.POSENET_INPUT_PARAMS['channel_increase']
    ).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = HeatMapLossBatch()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
    if rank == 0:
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)  # saving it in one process is sufficient.

    dist.barrier()  # Use a barrier() to make sure that process 1 loads the model after process 0 saves it.

    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}  # configure map_location properly
    ddp_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=map_location))

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    loss_fn = HeatMapLossBatch()
    loss_fn(outputs, labels).backward()
    optimizer.step()

    # Not necessary to use a dist.barrier() to guard the file deletion below
    # as the AllReduce ops in the backward pass of DDP already served as
    # a synchronization.

    if rank == 0:
        os.remove(CHECKPOINT_PATH)

    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn, args=(world_size,), nprocs=world_size, join=True)


# if __name__ == "__main__":
#     n_gpus = torch.cuda.device_count()
#     if n_gpus < 8:
#         print(f"Requires at least 8 GPUs to run, but got {n_gpus}.")
#     else:
#         run_demo(demo_basic, 8)
#         run_demo(demo_checkpoint, 8)
