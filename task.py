import os
import numpy as np
import torch
from torch import nn
from torch.nn import DataParallel
from utils.utils import make_input, make_output
from utils.read_config import ConfigYamlParserMPII
from models import PoseNet


class Trainer(nn.Module):
    """
    The wrapper module that will behave differetly for training or testing
    inference_keys specify the inputs for inference

    Args:
        model (torch.nn.Module):
        calc_loss (callable):
    """

    def __init__(self, model, calc_loss=None):
        super(Trainer, self).__init__()
        self.model = model
        self.calc_loss = calc_loss

    def forward(self, imgs, ground_truth=None):
        """

        Args:
            imgs (torch.Tensor): Batch of input images as tensor of shape (N, C, H, W).
            ground_truth (torch.Tensor): Labels for the corresponding input image .

        Returns:
            torch.Tensor or list[list[torch.Tensor], list[torch.Tensor]]: The output can be one of the following:
                - In inference or testing phase: Output of this method is tensor of shape ``(N, n_hourglass, C, (H+1)/4, (W+1)/4)``.
                - In training phase: Output of this method is list containg the following elements in given order:
                    - ``list`` of length ``1`` with each element as ``torch.Tensor`` of shape ``(N, n_hourglass, C, (H+1)/4, (W+1)/4)``.
                    - ``list`` of length ``1`` with each element as ``torch.Tensor`` of Shape ``(N, n_hourglass)``.
        """

        if not self.training:       # inference or testing phase
            return self.model(imgs)
        else:                       # training phase
            combined_heatmap_predictions = self.model(imgs)
            loss = self.calc_loss(combined_heatmap_predictions, ground_truth)
            return list([combined_heatmap_predictions]) + list([loss])


def make_pose_net(config):
    """
    Creates the neural network.

    Args:
        config (ConfigYamlParserMPII):

    Returns:
        callable:
    """

    train_params = config.NN_TRAINING_PARAMS                # dict
    inference_params = config.NN_INFERENCE_PARAMS           # dict

    def calc_loss(prediction, ground_truth):
        return pose_net.calc_loss(prediction, ground_truth)

    # create network object
    pose_net = PoseNet(n_hourglass=config.POSENET_INPUT_PARAMS['n_hourglass'],
                       in_channels=config.POSENET_INPUT_PARAMS['in_channels'],
                       out_channels=config.POSENET_INPUT_PARAMS['out_channels'],
                       channel_increase=config.POSENET_INPUT_PARAMS['channel_increase'])

    # dump network on gpu
    forward_net = DataParallel(pose_net.cuda())

    network_trainer = Trainer(forward_net, calc_loss)

    # optimizer, experiment setup
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, network_trainer.parameters()), train_params['learning_rate'])

    # experiment path
    # experiment_path = os.path.join('exp', configs['opt'].exp)
    # if configs['opt'].exp == 'pose' and configs['opt'].continue_exp is not None:
    #     experiment_path = os.path.join('exp', configs['opt'].continue_exp)
    # if not os.path.exists(experiment_path):
    #     os.mkdir(experiment_path)
    # logger = open(os.path.join(experiment_path, 'log'), 'a+')

    def make_train(batch_id, configuration, phase, **inputs):
        """

        Args:
            batch_id (int):
            configuration (dict):
            phase (str): `train` or `valid`
            **inputs: Keyword arguments.

        Returns:

        """
        configuration['batch_id'] = batch_id

        for i in inputs:
            try:
                inputs[i] = make_input(inputs[i])
            except:
                pass    # for last input, which is a string (id_)

        nnet = configuration['neural_network']
        """
        net (Trainer): 
        """

        nnet = nnet.train()
        if phase != 'inference':
            result = nnet(imgs=inputs['imgs'], ground_truth=inputs['ground_truth'])  # results is ``list[list[torch.Tensor], list[torch.Tensor]]``

            # num_loss = len(configuration['train']['loss'])  # num_loss = 1
            # losses = {i[0]: result[-num_loss + idx]*i[1] for idx, i in enumerate(config['train']['loss'])}
            losses = dict(
                combined_heatmap_loss=result[-1]
            )

            loss = 0
            toprint = '\n{}: '.format(batch_id)
            for i in losses:
                loss = loss + torch.mean(losses[i])
                my_loss = make_output(losses[i])

                if my_loss.size == 1:
                    toprint += ' {}: {}'.format(i, format(my_loss.mean(), '.8f'))
                else:
                    toprint += '\n{}'.format(i)
                    for j in my_loss:
                        toprint += ' {}'.format(format(j.mean(), '.8f'))
            # logger.write(toprint)
            # logger.flush()

            if phase == 'train':
                # optimizer = train_cfg['optimizer']
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if batch_id == config['train']['decay_iters']:
                # decrease the learning rate after decay # iterations
                for param_group in optimizer.param_groups:
                    param_group['lr'] = config['train']['decay_lr']
            return None
        else:
            out = {}
            net = net.eval()
            result = net(**inputs)
            if type(result) != list and type(result) != tuple:
                result = [result]
            out['preds'] = [make_output(i) for i in result]
            return out

    return make_train
