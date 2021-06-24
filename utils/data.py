"""
Website: http://human-pose.mpi-inf.mpg.de/#download
GitHub: https://github.com/princeton-vl/pytorch_stacked_hourglass/
"""

import sys
import os
import os.path as osp
import time
import h5py
import numpy as np
import cv2
from imageio import imread
import torch
import torch.utils.data
import utils
from utils.read_config import ConfigYamlParserMPII

# MAX_ROTATION_ANGLE = 30.   # Maximum rotation for data augmentation by rotation of image. (in degree)
# SCALE_RANGE = dict(min=0.75, max=1.75)  # Image scaling factor range for data augmentation
# REFERENCE_PIXEL_SIZE = 200
# """
# int: Reference pixel size of Person's image. MPII dataset stores the scale with reference to `200 pixel` size.
#
# References:
#     * MPII website: http://human-pose.mpi-inf.mpg.de/#download
# """
#
# CONFIG_FILE_PATH = osp.abspath("../config.yaml")
# assert osp.exists(CONFIG_FILE_PATH), "Configuration file not found. Check if the config.yaml file exists in the directory."
#
# with open(CONFIG_FILE_PATH) as f:
#     config = yaml.load(f, Loader=yaml.FullLoader)
#
# base_dir = osp.join(osp.dirname(CONFIG_FILE_PATH), config['data']['base'])
# assert osp.exists(base_dir), f"base directory {base_dir} not found. Check {CONFIG_FILE_PATH} file."
#
# train_annotation_file = osp.join(base_dir, config['data']['train']['annotations'])
# assert osp.exists(train_annotation_file), f"Training annotation file {train_annotation_file} not found. Check {CONFIG_FILE_PATH} file."
#
# validation_annotation_file = osp.join(base_dir, config['data']['validation']['annotations'])
# assert osp.exists(validation_annotation_file), f"Validation annotation file {validation_annotation_file} not found. Check {CONFIG_FILE_PATH} file."
#
# image_dir = osp.join(base_dir, config['data']['images'])
# assert osp.exists(image_dir), f"Image directory {image_dir} not found. Check {CONFIG_FILE_PATH} file."
#
# # Part reference
# parts = {'mpii': ['rank', 'rkne', 'rhip', 'lhip', 'lkne', 'lank', 'pelv', 'thrx', 'neck', 'head', 'rwri', 'relb', 'rsho', 'lsho', 'lelb', 'lwri']}
# flipped_parts = {'mpii': [5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 15, 14, 13, 12, 11, 10]}
# part_pairs = {'mpii': [[0, 5], [1, 4], [2, 3], [6], [7], [8], [9], [10, 15], [11, 14], [12, 13]]}
# pair_names = {'mpii': ['ankle', 'knee', 'hip', 'pelvis', 'thorax', 'neck', 'head', 'wrist', 'elbow', 'shoulder']}


def _is_array_like(obj_):
    """
    Check if the input object is like an array or not.

    Args:
        obj_ (object): Input object.

    Returns:
        bool: `True` if the input object has ``__iter__`` and ``__len__`` attributes.
    """
    return hasattr(obj_, '__iter__') and hasattr(obj_, '__len__')


class MPIIAnnotationHandler:
    """
    Class for MPII Dataset Annotation handling. Here only training and validation annotations are handled.

    TODO: Handle testing annotations in this class. (MAY BE...)

    Args:
        train_annotation_file (str): Path to annotations (a.k.a. labels) file for training set.
        validation_annotation_file (str): Path to annotations (a.k.a. labels) file for validation set.
        image_dir (str): Path to directory containing the MPII dataset images.
        keypoint_info (dict): Information about the human-pose keypoints. Taken from ``config.yaml`` file of MPII dataset. Information under ```['data']['MPII']['parts']``` in ``config.yaml``.

    References:
        * https://dbcollection.readthedocs.io/en/latest/datasets/mpii_pose.html
    """

    def __init__(self, train_annotation_file, validation_annotation_file, image_dir, keypoint_info):
        print('loading data...')
        tic = time.time()

        train_file = h5py.File(train_annotation_file, 'r')
        validation_file = h5py.File(validation_annotation_file, 'r')
        self.image_dir = image_dir
        self.keypoint_info = keypoint_info

        train_center = train_file['center'][()]  # center coordinates (x, y) of a single person detection
        train_scale = train_file['scale'][()]
        train_part = train_file['part'][()]
        train_visible = train_file['visible'][()]
        train_normalize = train_file['normalize'][()]

        self.n_train_samples = len(train_center)
        """
        int: Number of training samples available in dataset.
        """

        train_imgname = [None] * self.n_train_samples
        for i in range(self.n_train_samples):
            train_imgname[i] = train_file['imgname'][i].decode('UTF-8')

        validation_center = validation_file['center'][()]
        validation_scale = validation_file['scale'][()]
        validation_part = validation_file['part'][()]
        validation_visible = validation_file['visible'][()]
        validation_normalize = validation_file['normalize'][()]

        self.n_validation_samples = len(validation_center)
        """
        int: Number of validation samples available in dataset.
        """

        validation_imgname = [None] * self.n_validation_samples
        for i in range(self.n_validation_samples):
            validation_imgname[i] = validation_file['imgname'][i].decode('UTF-8')

        self.center = np.append(train_center, validation_center, axis=0)
        self.scale = np.append(train_scale, validation_scale)
        self.part = np.append(train_part, validation_part, axis=0)
        self.visible = np.append(train_visible, validation_visible, axis=0)
        self.normalize = np.append(train_normalize, validation_normalize)
        self.imgname = train_imgname + validation_imgname

        print('Done (t={:0.2f}s)'.format(time.time() - tic))

    def get_annotation(self, idx):
        """
        Returns h5 file for train or val set

        Args:
            idx (int): Index of the data.

        Returns:
            tuple: Tuple containing (image_filename, part_keypoints, visibility_flag, center_coordinates, scale_factor, normalize_factor) from MPII annotations.

        Notes:
            - ``part`` is an array of human-pose keypoint coordinates in the image.
        """
        return (self.imgname[idx],
                self.part[idx],
                self.visible[idx],
                self.center[idx],
                self.scale[idx],
                self.normalize[idx]
                )

    def get_imgpath(self, img_name):
        """
        Get image file path from image file name.

        Args:
            img_name (str): Image file name.

        Returns:
            str: complete path to the given image file.
        """
        return osp.join(self.image_dir, img_name)

    def get_length(self):
        """

        Returns:
            tuple[int, int]: tuple of elements ``(sample_count_in_training_set, sample_count_in_validation_set)``.
        """
        return self.n_train_samples, self.n_validation_samples

    def split_data(self):
        """
        Split the MPII dataset into training and validation set. Returns index for train and validation imgs.
        Indices for validation images starts after that of train images so that load_image can tell them apart

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: Tuple containing:
                - numpy.ndarray: training data indices with shape (N_train,)
                - numpy.ndarray: validation data indices with shape (N_validation,)
        """
        train = [i for i in range(self.n_train_samples)]
        valid = [i + self.n_train_samples for i in range(self.n_validation_samples)]
        return np.array(train), np.array(valid)

    def get_image(self, idx):
        """
        Returns the image at the provided index.

        Args:
            idx (int): Index of the data sample.

        Returns:
            numpy.ndarray: Returns a numpy array, which comes with a dict of meta data at its ‘meta’ attribute.

        References:
            imageio.imread: https://imageio.readthedocs.io/en/stable/userapi.html#imageio.imread
        """
        imgname, __, __, __, __, __ = self.get_annotation(idx)
        path = osp.join(self.image_dir, imgname)
        img = imread(path)
        return img

    def load_image(self, path):
        """
        Load image from path.

        Args:
            path (str): Image file path.

        Returns:
            numpy.ndarray: Returns a numpy array, which comes with a dict of meta data at its ‘meta’ attribute.

        References:
            imageio.imread: https://imageio.readthedocs.io/en/stable/userapi.html#imageio.imread
        """
        return imread(path)

    def get_imgpath_from_idx(self, idx):
        """
        Returns the path of image file corresponding to provided data index.

        Args:
            idx (int): Index of the data sample.

        Returns:
            str: Path of image file.

        """
        imgname, __, __, __, __, __ = self.get_annotation(idx)
        path = osp.join(self.image_dir, imgname)
        return path

    def get_kps(self, idx):
        """
        Get Key Points on the body.

        Args:
            idx (int): Index of the data sample.

        Returns:
            numpy.ndarray: Keypoints corresponding to index ``idx`` with shape ``(1, 16, 3)`` with each row of idx ``0`` containing ``(x, y, visibility_flag)``.
        """
        __, part, visible, __, __, __ = self.get_annotation(idx)
        kp2 = np.insert(part, 2, visible, axis=1)
        kps = np.zeros((1, 16, 3))
        kps[0] = kp2
        return kps

    def merge_keypoints_visibility(self, kp, visibility):
        """
        Args:
            kp (numpy.ndarray): keypoints corresponding to datasample with shape ``(16, 2)``.
            visibility (numpy.ndarray): visibility corresponding to datasample with shape ``(16,)``.

        Returns:
            numpy.ndarray: Keypoints with shape ``(1, 16, 3)`` with each row of idx ``0`` containing ``(x, y, visibility_flag)``.
        """
        kp2 = np.insert(kp, 2, visibility, axis=1)
        kps = np.zeros((1, 16, 3))
        kps[0] = kp2
        return kps

    def get_normalized(self, idx):
        """
        Returns normalized value for given index from MPII dataset.

        Args:
            idx (int): Index of the data sample.

        Returns:
            float:
        """
        __, __, __, __, __, n = self.get_annotation(idx)
        return n

    def get_center(self, idx):
        __, __, __, c, __, __ = self.get_annotation(idx)
        return c

    def get_scale(self, idx):
        __, __, __, __, s, __ = self.get_annotation(idx)
        return s


class GenerateHeatmap:
    """
    Generate heatmap for data. Creates a callable object which returns the heatmap based on the input arguments.

    Args:
        output_resolution (int): Output resolution. Default is ``64``.
        num_parts (int): Number of parts. This number depends on the dataset. MPII has ``16`` parts. Refer the labels of the parts in ``torch_shnet.utils.data.parts['mpii']`` attribute of the dictionary for more details.

    Examples:
        >>> gen_heatmap = GenerateHeatmap( output_resolution=64, num_parts=16)
        >>> keypoints = np.random.rand(1, 16, 3)
        >>> heat_map = gen_heatmap(keypoints)   # For MPII dataset this is (1, 16, 3)

    """

    def __init__(self, output_resolution=64, num_parts=16):
        self.output_resolution = output_resolution
        self.num_parts = num_parts

        sigma = self.output_resolution / 64.
        self.sigma = sigma
        self.gaussian, self.x_center, self.y_center = GenerateHeatmap.create_gaussian_kernel(sigma=self.sigma)

    @staticmethod
    def create_gaussian_kernel(sigma=1.0, size=None):
        """
        Method to create a gaussian kernel.

        Args:
            sigma (float): Standard deviation of gaussian kernel.
            size (int): Size of gaussian kernel. Default is ``None``.

        Returns:
            tuple[numpy.ndarray, int, int]: Tuple of length ``3`` containing:
                - numpy.ndarray: Gaussian Kernel of shape (size, size). Default shape is ``(9, 9)``.
                - int: Gaussian kernel center (mean) along x-axis.
                - int: Gaussian kernel center (mean) along y-axis.
        """

        if size is None:
            size = 6 * sigma + 3
            x_center = y_center = 3 * sigma + 1
        else:
            x_center = y_center = int((size-1)/2)

        assert int(size) % 2 == 1, f"Size {size} must be an odd number."

        x = np.arange(start=0, stop=size, step=1, dtype=float)
        y = x[:, None]

        gaussian = np.exp(- ((x - x_center) ** 2 + (y - y_center) ** 2) / (2 * sigma ** 2))
        return gaussian, x_center, y_center

    def __call__(self, keypoints):
        """
        Call object by passing the keypoints.

        Args:
            keypoints (numpy.ndarray): Keypoint array of shape ``(M, N, 2)``. ``M=1`` and ``N=16`` for MPII dataset.

        Returns:
            numpy.ndarray: Heatmap of keypoints with shape ``(num_parts, output_resolution, output_resolution)``.

        """
        heatmaps_shape = (self.num_parts, self.output_resolution, self.output_resolution)
        heatmaps = np.zeros(shape=heatmaps_shape, dtype=np.float32)

        sigma, xc, yc = self.sigma, self.x_center, self.y_center
        kernel_size = self.gaussian.shape[0]

        for p in keypoints:
            for idx, pt in enumerate(p):
                if pt[0] > 0:
                    x, y = int(pt[0]), int(pt[1])

                    if x < 0 or y < 0 or x >= self.output_resolution or y >= self.output_resolution:
                        continue

                    ul = int(x - xc), int(y - yc)                                           # upper left corner
                    br = int(x + (kernel_size-xc)), int(y + (kernel_size-yc))               # bottom right corner

                    # overlapping gaussian-kernel indices with the heatmap
                    c, d = max(0, -ul[0]), min(br[0], self.output_resolution) - ul[0]       # (x_min, x_max)
                    a, b = max(0, -ul[1]), min(br[1], self.output_resolution) - ul[1]       # (y_min, y_max)

                    # (min, max) limits of gaussian-kernel on the heatmap along x-axis and y-axis
                    cc, dd = max(0, ul[0]), min(br[0], self.output_resolution)              # (x_min, x_max)
                    aa, bb = max(0, ul[1]), min(br[1], self.output_resolution)              # (y_min, y_max)

                    heatmaps[idx, aa:bb, cc:dd] = np.maximum(heatmaps[idx, aa:bb, cc:dd], self.gaussian[a:b, c:d])
        return heatmaps


class MPIIDataset(torch.utils.data.Dataset):
    """
    MPII dataset handling.

    Args:
        input_resolution (int): Input resolution of image samples.
        output_resolution (int): Output resolution of heatmaps
        num_parts (int): Number of parts on the body to recognize. For MPII dataset, this value is ``16``.
        indices (numpy.ndarray): Indices of the dataset.
        mpii_annotation_handle (MPIIAnnotationHandler): Object to handle MPII dataset annotations.
        reference_image_size (int): Reference size of the image according to what scaling was done. This value is ``200 px`` for MPII dataset. The default is `200`.
        image_scale_factor_range (tuple[float, float]): Tuple of length ``2`` as (image_scaling_factor_lower_limit, image_scaling_factor_upper_limit). Default is ``(0.75, 1.75)``.
        image_color_jitter_probability (float): Probability of random jittering of image sample. Default is `0.5`.
        hue_max_delta (float): Maximum purturbabtion in the hue of image. Default is ``0.2``.
        saturation_min_delta (float): Minimum saturation factor. Default is ``0.2``.
        brightness_max_delta (float): Maximum possible change in image brightness. Default is ``0.3``.
        contrast_min_delta (float): minimum possible change in contrast of image. Default is ``0.5``.

    """
    def __init__(self,
                 input_resolution,
                 output_resolution,
                 num_parts,
                 indices,
                 mpii_annotation_handle,
                 reference_image_size=200,
                 max_rotation_angle=30.,
                 image_scale_factor_range=(0.75, 1.75),
                 image_color_jitter_probability=0.5,
                 hue_max_delta=0.2,
                 saturation_min_delta=0.5,
                 brightness_max_delta=0.3,
                 contrast_min_delta=0.5):

        assert image_scale_factor_range[0] > 0 and image_scale_factor_range[1] > 0
        assert image_scale_factor_range[0] <= image_scale_factor_range[1]
        assert 0. < image_color_jitter_probability < 1.0

        self.input_resolution = input_resolution
        self.output_resolution = output_resolution
        self.indices = indices
        self.reference_image_size = reference_image_size
        self.mpii_annotation_handle = mpii_annotation_handle
        self.max_rotation_angle = max_rotation_angle
        self.image_scaling_factor = {"min": image_scale_factor_range[0], "max": image_scale_factor_range[1]}
        self.image_color_jitter_probability = image_color_jitter_probability
        self.hue_max_delta = hue_max_delta
        self.saturation_min_delta = saturation_min_delta
        self.brightness_max_delta = brightness_max_delta
        self.contrast_min_delta = contrast_min_delta

        self.generate_heatmap = GenerateHeatmap(self.output_resolution, num_parts)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.load_image(self.indices[idx % len(self.indices)])

    def load_image(self, idx):
        """
        Load the image at the given index `idx` of the dataset.

        Args:
            idx (int): index of the data sample.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: Tuple of shape ``(2,)`` containing:
                - numpy.ndarray: Input image of shape ``(H_in, W_in, C_in)``. Mostly as ``H_in=W_in=256`` and channels ``C_in=3``. The input image is normalized to have pixel values between ``[0, 1]``. The content of the numpy array is of ``np.float32``.
                - numpy.ndarray: Output heatmaps containing keypoints on the person body of shape ``(Parts, H_out, W_out)``. Likely to be of shape ``(16, 64, 64)``. The content of the numpy array is of ``np.float32``.
        """
        # load image and relevant data for label
        image_filename, part_keypoints, visibility_flag, c, s, normalize = self.mpii_annotation_handle.get_annotation(idx)
        path = self.mpii_annotation_handle.get_imgpath(image_filename)
        orig_img = self.mpii_annotation_handle.load_image(path)
        orig_keypoints = self.mpii_annotation_handle.merge_keypoints_visibility(part_keypoints, visibility_flag)

        kptmp = orig_keypoints.copy()

        # crop the image to extract the image of the person
        cropped = utils.utils.crop(img=orig_img, center=c, scale=s, resolution=(self.input_resolution, self.input_resolution), rotation=0)

        # Transform pixel locations of keypoints to desired input image resolution
        for i in range(np.shape(orig_keypoints)[1]):
            if orig_keypoints[0, i, 0] > 0:
                orig_keypoints[0, i, :2] = utils.utils.transform(pt=orig_keypoints[0, i, :2], center=c, scale=s, resolution=(self.input_resolution, self.input_resolution))

        keypoints = np.copy(orig_keypoints)

        height, width = cropped.shape[0:2]   # height and width of cropped image

        center = np.array((width / 2, height / 2))   # center of cropped image

        scale = max(height, width) / self.reference_image_size          # scale cropped image.

        aug_rot = (np.random.random() * 2 - 1) * self.max_rotation_angle    # augmentation - random rotation (in degrees) of cropped image

        aug_scale = np.random.random() * (self.image_scaling_factor['max'] - self.image_scaling_factor['min']) + self.image_scaling_factor['min']     # augmentation - random scaling of cropped image

        scale *= aug_scale          # scale augmentation

        mat = utils.utils.get_transform(center, scale, (self.input_resolution, self.input_resolution), aug_rot)[:2]  # transformation matrix - transform the image as per the augmentation

        inp = cv2.warpAffine(cropped, mat, (self.input_resolution, self.input_resolution)).astype(np.float32) / 255   # image (data) augmentation

        # transform the output keypoints to be included in heatmaps as per the augmentation
        mat_mask = utils.utils.get_transform(center, scale, (self.output_resolution, self.output_resolution), aug_rot)[:2]
        keypoints[:, :, 0:2] = utils.utils.kpt_affine(keypoints[:, :, 0:2], mat_mask)

        # random color jitter
        if np.random.rand() > self.image_color_jitter_probability:
            flipped_keypoint_ids = self.mpii_annotation_handle.keypoint_info['parts']['flipped_ids']

            inp = self.random_color_jitter(inp)
            inp = inp[:, ::-1]
            keypoints = keypoints[:, flipped_keypoint_ids]
            keypoints[:, :, 0] = self.output_resolution - keypoints[:, :, 0]
            orig_keypoints = orig_keypoints[:, flipped_keypoint_ids]
            orig_keypoints[:, :, 0] = self.input_resolution - orig_keypoints[:, :, 0]

        # set keypoints to 0 when were not visible initially (so heatmap all 0s)
        for i in range(np.shape(orig_keypoints)[1]):
            if kptmp[0, i, 0] == 0 and kptmp[0, i, 1] == 0:
                keypoints[0, i, 0] = 0
                keypoints[0, i, 1] = 0
                orig_keypoints[0, i, 0] = 0
                orig_keypoints[0, i, 1] = 0

        # generate heatmaps on output resolution
        heatmaps = self.generate_heatmap(keypoints)

        return inp.astype(np.float32), heatmaps.astype(np.float32)

    def random_color_jitter(self, data):
        """
        Color jittering of input image.

        Args:
            data (numpy.ndarray): Image with pixel values scaled between ``[0, 1]`` of shape ``(H, W, C)``.

        Returns:
            numpy.ndarray: Augmented image sample with same as input shape.
        """

        # image from RGB to HSV
        data = cv2.cvtColor(data, cv2.COLOR_RGB2HSV)

        # hue augmentation
        delta = (np.random.random() * 2 - 1) * self.hue_max_delta
        data[:, :, 0] = np.mod(data[:, :, 0] + (delta * 360 + 360.), 360.)

        # saturation augmentation
        delta_sature = np.random.random() + self.saturation_min_delta
        data[:, :, 1] *= delta_sature
        data[:, :, 1] = np.maximum(np.minimum(data[:, :, 1], 1), 0)

        # image from HSV to RGB
        data = cv2.cvtColor(data, cv2.COLOR_HSV2RGB)

        # adjust brightness
        delta = (np.random.random() * 2 - 1) * self.brightness_max_delta
        data += delta

        # adjust contrast
        mean = data.mean(axis=2, keepdims=True)
        data = (data - mean) * (np.random.random() + self.contrast_min_delta) + mean
        data = np.minimum(np.maximum(data, 0), 1)

        return data


def init(config):
    """
    Initialize data loader

    Args:
        config (ConfigYamlParserMPII): Configuration parsed using ``config.yaml`` file for MPII dataset.

    Returns:
        types.LambdaType:
    """
    current_path = osp.dirname(os.path.abspath(__file__))
    sys.path.append(current_path)

    batch_size = int(config.NN_TRAINING_PARAMS['batch_size'])
    input_resolution = int(config.NN_TRAINING_PARAMS['input_resolution'])
    output_resolution = int(config.NN_TRAINING_PARAMS['output_resolution'])
    num_workers = int(config.NN_TRAINING_PARAMS['num_workers'])

    num_parts = int(config.PARTS['max_count'])

    mpii_annotation_handle = MPIIAnnotationHandler(
        train_annotation_file=config.TRAINING_ANNOTATION_FILE,
        validation_annotation_file=config.VALIDATION_ANNOTATION_FILE,
        image_dir=config.IMAGE_DIR,
        keypoint_info=config.PARTS)

    training_data_indices, validation_data_indices = mpii_annotation_handle.split_data()

    __data_aug = config.NN_TRAINING_PARAMS['data_augmentation']
    __img_sf = __data_aug['image_scale_factor']
    image_scale_factor_range = (__img_sf['min'], __img_sf['max'])
    dataset = {key: MPIIDataset(
                        input_resolution=input_resolution,
                        output_resolution=output_resolution,
                        num_parts=num_parts,
                        indices=indices,
                        mpii_annotation_handle=mpii_annotation_handle,
                        reference_image_size=config.REFERENCE_IMAGE_SIZE,
                        max_rotation_angle=__data_aug['rotation_angle_max'],
                        image_scale_factor_range=image_scale_factor_range,
                        image_color_jitter_probability=__data_aug['image_color_jitter_probability'],
                        hue_max_delta=__data_aug['hue_max_delta'],
                        saturation_min_delta=__data_aug['saturation_min_delta'],
                        brightness_max_delta=__data_aug['brightness_max_delta'],
                        contrast_min_delta=__data_aug['contrast_min_delta']
                    ) for key, indices in zip(['train', 'valid'], [training_data_indices, validation_data_indices])}

    loaders = {}
    for key in dataset:
        loaders[key] = torch.utils.data.DataLoader(dataset[key], batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)

    def gen(phase):
        """
        Args:
            phase (str): Phase can take one value out of ('train', 'valid') for training and validation respectively.

        Yields:
            dict: Dictionary containing keys 'imgs' and 'heatmaps' corresponding to images and heatmaps returned in each batch.
        """
        iters = config.NN_TRAINING_PARAMS['{}_iterations'.format(phase)]
        loader = loaders[phase].__iter__()
        for i in range(iters):
            try:
                imgs, heatmaps = next(loader)
            except StopIteration:   # to avoid no data provided by dataloader
                loader = loaders[phase].__iter__()
                imgs, heatmaps = next(loader)
            yield {
                'imgs': imgs,               # cropped and augmented
                'heatmaps': heatmaps,       # based on keypoints. 0 if not in img for joint
            }
    return lambda key: gen(key)
