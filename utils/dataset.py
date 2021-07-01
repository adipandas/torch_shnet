import numpy as np
import torch
from cv2 import warpAffine
from imageio import imread
from torch import tensor
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ColorJitter, Compose as transformCompose
import utils.helpers as utils
from utils.annotation_handler import MPIIAnnotationHandler
from utils.heatmap_generator import GenerateHeatmap


class MPIIDataset(Dataset):
    """
    MPII dataset handling.

    Args:
        indices (numpy.ndarray): Indices of the dataset.
        mpii_annotation_handle (MPIIAnnotationHandler): Object to handle MPII dataset annotations.
        horizontally_flipped_keypoint_ids (list or tuple): List of IDs of the human-pose keypoints flipped horizontally in the image. Taken from ``config.yaml`` file of MPII dataset. Information under ```['data']['MPII']['parts']['flipped_ids']``` in ``config.yaml``.
        input_resolution (int): Desired Input resolution of image samples to pass in neural network. Default is ``256`` pixel.
        output_resolution (int): Desired Output resolution of heatmaps to form ground truth of neural network. Default is ``64``pixel.
        num_parts (int): Number of parts on the body to recognize. For MPII dataset, this value is ``16``.
        reference_image_size (int): Reference size of the image according to what scaling was done. This value is ``200 px`` for MPII dataset. The default is `200`.
        image_scale_factor_range (tuple[float, float]): Tuple of length ``2`` as (image_scaling_factor_lower_limit, image_scaling_factor_upper_limit). Default is ``(0.75, 1.75)``.
        image_color_jitter_probability (float): Probability of random jittering of image sample. Default is `0.5`.
        image_horizontal_flip_probability (float): Probability of flipping the image horizontally. Default is `0.5`.
        hue_max_delta (float): Maximum purturbabtion in the hue of image. Default is ``0.2``.
        saturation_min_delta (float): Minimum saturation factor. Default is ``0.2``.
        brightness_max_delta (float): Maximum possible change in image brightness. Default is ``0.3``.
        contrast_min_delta (float): minimum possible change in contrast of image. Default is ``0.5``.

    Notes:
        * This class is to be used with ``torch.utils.data.DataLoader``.
        * __getitem__ returns the following in given order:
            - image: Torch Tensor with shape (N, 3, H, W)
            - heatmaps: Torch Tensor with shape (N, num_parts, H, W)

    References:
        * Website: http://human-pose.mpi-inf.mpg.de/#download
        * GitHub: https://github.com/princeton-vl/pytorch_stacked_hourglass/

    """

    def __init__(self,
                 indices,
                 mpii_annotation_handle,
                 horizontally_flipped_keypoint_ids,
                 input_resolution=256,
                 output_resolution=64,
                 num_parts=16,
                 reference_image_size=200,
                 max_rotation_angle=30.,
                 image_scale_factor_range=(0.75, 1.75),
                 image_color_jitter_probability=0.5,
                 image_horizontal_flip_probability=0.5,
                 hue_max_delta=0.2,
                 saturation_min_delta=0.5,
                 brightness_max_delta=0.3,
                 contrast_min_delta=0.5):

        assert image_scale_factor_range[0] > 0 and image_scale_factor_range[1] > 0
        assert image_scale_factor_range[0] <= image_scale_factor_range[1]
        assert 0. < image_color_jitter_probability < 1.0
        assert 0. < image_horizontal_flip_probability < 1.0
        self.indices = indices
        self.mpii_annotation_handle = mpii_annotation_handle
        self.horizontally_flipped_keypoint_ids = horizontally_flipped_keypoint_ids
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution
        self.reference_image_size = reference_image_size
        self.max_rotation_angle = max_rotation_angle
        self.image_scaling_factor = {"min": image_scale_factor_range[0], "max": image_scale_factor_range[1]}
        self.image_color_jitter_probability = image_color_jitter_probability
        self.image_horizontal_flip_probability = image_horizontal_flip_probability
        self.hue_max_delta = hue_max_delta
        self.saturation_min_delta = saturation_min_delta
        self.brightness_max_delta = brightness_max_delta
        self.contrast_min_delta = contrast_min_delta

        self.transforms = transformCompose([
            ColorJitter(brightness=brightness_max_delta, contrast=contrast_min_delta, saturation=saturation_min_delta, hue=hue_max_delta)
        ])

        self.generate_heatmap = GenerateHeatmap(self.output_resolution, num_parts)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx % len(self.indices)]
        in_res = (self.input_resolution, self.input_resolution)             # input height, width
        out_res = (self.output_resolution, self.output_resolution)          # output height, width
        s_min, s_max = self.image_scaling_factor['min'], self.image_scaling_factor['max']

        imgpath, kp, c, s = self.mpii_annotation_handle[idx]
        image = imread(imgpath)                                             # load image with shape ``(H, W, C)``.
        o_kp = kp.copy()                                                    # copy of original keypoint coordinates

        # transform image and original keypoints to desired input resolution and scale
        image, input_keypoints = utils.transform_MPII_image_keypoints(image, kp, c, s, in_res)
        height, width = image.shape[0:2]                            # height and width of input image
        center = np.array((width / 2, height / 2))                  # center of input image
        scale = max(height, width) / self.reference_image_size      # scale input image.

        # rotation and scaling augmentation of input image
        scale *= (np.random.random() * (s_max - s_min) + s_min)     # scale factor augmentation
        augment_rotation_angle = (np.random.random() * 2. - 1.) * self.max_rotation_angle  # random rotation angle - augmentation

        # augmented transformation matrix for input and output data
        input_transformation_matrix = utils.get_transform(center, scale, in_res, augment_rotation_angle, self.reference_image_size)[:2]
        output_transformation_matrix = utils.get_transform(center, scale, out_res, augment_rotation_angle, self.reference_image_size)[:2]

        # augment input image
        image = warpAffine(image, input_transformation_matrix, in_res).astype(np.float32) / 255.    # image (data) augmentation

        # transform the output keypoints to be included in heatmaps as per the augmentation
        output_keypoints = input_keypoints.copy()
        output_keypoints[:, :, 0:2] = utils.kpt_affine(output_keypoints[:, :, 0:2], output_transformation_matrix)

        # transform the input keypoints as per the augmentation - do it after the output_transformation
        input_keypoints[:, :, 0:2] = utils.kpt_affine(input_keypoints[:, :, 0:2], input_transformation_matrix)

        if np.random.rand() < self.image_horizontal_flip_probability:
            image = utils.image_horizontal_flip(image)
            input_keypoints = utils.keypoints_horizontal_flip(input_keypoints, self.input_resolution, self.horizontally_flipped_keypoint_ids)
            output_keypoints = utils.keypoints_horizontal_flip(output_keypoints, self.output_resolution, self.horizontally_flipped_keypoint_ids)

        # set keypoints to 0 when were not visible initially (so heatmap all 0s)
        for i in range(np.shape(input_keypoints)[1]):
            if o_kp[0, i, 0] == 0 or o_kp[0, i, 1] == 0:
                input_keypoints[0, i, 0:2] *= 0
                output_keypoints[0, i, 0:2] *= 0

        heatmaps = self.generate_heatmap(output_keypoints)  # generate heatmaps on output resolution

        if self.transforms:
            image = image.transpose((2, 0, 1)).copy()  # if copy is not done, torch throws exception
            image = torch.from_numpy(image)         # image to torch tensor
            image = self.transforms(image)          # color jitter in input image; create torch tensors for input and output
        heatmaps = tensor(heatmaps)

        return image, heatmaps  # , input_keypoints, output_keypoints
