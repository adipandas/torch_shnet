"""
* Website: http://human-pose.mpi-inf.mpg.de/#download
*  GitHub: https://github.com/princeton-vl/pytorch_stacked_hourglass/
"""

import os.path as osp
import time
import h5py
import numpy as np
from cv2 import warpAffine, cvtColor, COLOR_HSV2RGB, COLOR_RGB2HSV
from imageio import imread
from torch import tensor
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import utils.helpers as utils


class MPIIAnnotationHandler:
    """
    Class for MPII Dataset Annotation handling. Here only training and validation annotations are handled.

    TODO: Handle testing annotations in this class. (MAY BE...)

    Attributes:
        center (numpy.ndarray): Center points of the person bounding boxes in MPII dataset with shape ``(data_length, 2)``.
        scale (numpy.ndarray): Scale factor corresponding to image size with respect to a reference value (Reference value is 200 pixel for MPII dataset). Shape ``(data_length,)``.
        part (numpy.ndarray): Keypoint coordinates corresponding to each sample. Shape ``(data_length, 16, 2)``.
        visible (numpy.ndarray): Visibility flag corresponding to each keypoint in the image sample. Shape ``(data_length, 16, 1)``.
        normalize (numpy.ndarray): Shape ``(data_length,)``.
        image_filename (list[str]): Name of image file corresponding to each sample. Length ``data_length``.
        n_train_samples (int): Number of samples in the training annotation file.
        n_validation_samples (int): Number of samples in the validation annotation file.
        image_dir (str): Path to directory containing the MPII dataset images.
        keypoint_info (dict): Information about the human-pose keypoints. Taken from ``config.yaml`` file of MPII dataset. Information under ```['data']['MPII']['parts']``` in ``config.yaml``.

    Args:
        training_annotation_file (str): Path to annotations (a.k.a. labels) file for training set in ``.h5`` format.
        validation_annotation_file (str): Path to annotations (a.k.a. labels) file for validation set in ``.h5`` format.
        image_dir (str): Path to directory containing the MPII dataset images.
        keypoint_info (dict): Information about the human-pose keypoints. Taken from ``config.yaml`` file of MPII dataset. Information under ```['data']['MPII']['parts']``` in ``config.yaml``.

    References:
        * https://dbcollection.readthedocs.io/en/latest/datasets/mpii_pose.html
    """

    def __init__(self, training_annotation_file, validation_annotation_file, image_dir, keypoint_info):
        print('loading data...')
        tic = time.time()

        self.image_dir = image_dir
        self.keypoint_info = keypoint_info

        (self.center, self.scale, self.part, self.visible, self.normalize, self.image_filename, self.n_train_samples, self.n_validation_samples) = self.__load_training_and_validation_annotation_file(training_annotation_file, validation_annotation_file)

        print('Done (t={:0.2f}s)'.format(time.time() - tic))

    def __load_annotations(self, annotation_file):
        """

        Args:
            annotation_file (str): Path to ``.h5`` annotation file.

        Returns:
            tuple: Tuple containing following elements in given order:
                - center (numpy.ndarray): Shape ``(data_length, 2)``.
                - scale (numpy.ndarray): Shape ``(data_length,)``.
                - part (numpy.ndarray): Shape ``(data_length, 16, 2)``.
                - visible (numpy.ndarray): Shape ``(data_length, 16, 1)``.
                - normalize (numpy.ndarray): Shape ``(data_length,)``.
                - data_length (int): Number of samples in the data.
                - filename (list[str]): Length ``data_length``.

        """
        data = h5py.File(annotation_file, 'r')

        center = data['center'][()]  # center coordinates (x, y) of a single person detection
        scale = data['scale'][()]
        part = data['part'][()]
        visible = data['visible'][()]
        normalize = data['normalize'][()]

        data_length = len(center)

        filename = [None] * data_length
        for i in range(data_length):
            filename[i] = data['imgname'][i].decode('UTF-8')

        return center, scale, part, visible, normalize, data_length, filename

    def __load_training_and_validation_annotation_file(self, training_annotation_file, validation_annotation_file):
        """
        Load training and validation annotation files.

        Args:
            training_annotation_file (str): Path to training annotation file (`.h5` format).
            validation_annotation_file (str): Path to validation annotation file (`.h5` format).

        Returns:
            tuple: Tuple containing following elements in given order:
                - center (numpy.ndarray): Shape ``(data_length, 2)``.
                - scale (numpy.ndarray): Shape ``(data_length,)``.
                - part (numpy.ndarray): Shape ``(data_length, 16, 2)``.
                - visible (numpy.ndarray): Shape ``(data_length, 16, 1)``.
                - normalize (numpy.ndarray): Shape ``(data_length,)``.
                - filename (list[str]): Length ``data_length``.
                - training_data_length (int): Number of samples in the training annotation file.
                - validation_data_length (int): Number of samples in the validation annotation file.

        Notes:
            * ``data_length = training_data_length + validation_data_length``

        """
        tcenter, tscale, tpart, tvisible, tnormalize, tdata_length, tfilename = self.__load_annotations(training_annotation_file)
        vcenter, vscale, vpart, vvisible, vnormalize, vdata_length, vfilename = self.__load_annotations(validation_annotation_file)

        center = np.append(tcenter, vcenter, axis=0)
        scale = np.append(tscale, vscale)
        part = np.append(tpart, vpart, axis=0)
        visible = np.append(tvisible, vvisible, axis=0)
        normalize = np.append(tnormalize, vnormalize)
        filename = tfilename + vfilename

        training_data_length = tdata_length
        validation_data_length = vdata_length

        return center, scale, part, visible, normalize, filename, training_data_length, validation_data_length

    def __get_image_file_path(self, image_file_name):
        """
        Get image file path from image file name.

        Args:
            image_file_name (str): Image file name.

        Returns:
            str: complete path to the given image file.
        """
        return osp.join(self.image_dir, image_file_name)

    def get_annotation(self, idx, full_path=True):
        """
        Returns data from h5 file for train or val set corresponding to given index ``idx``.

        Args:
            idx (int): Index of the data.
            full_path (bool): If ``True``, returns the full path of the sample image file. If ``false``, returns ONLY the image file name and NOT the full path. Default is ``True``.

        Returns:
            tuple: Tuple containing MPII dataset annotation in given order:
                - image_filename
                - keypoints (numpy.ndarray): Keypoints corresponding to index ``idx`` with shape ``(1, 16, 3)`` with each row of idx ``0`` containing ``(x, y, visibility_flag)``.
                - visibility_flag
                - center_coordinates
                - scale_factor
                - normalize_factor

        Notes:
            - ``part`` is an array of human-pose keypoint coordinates in the image.
        """
        image_filename = self.image_filename[idx]
        if full_path:
            image_filename = self.__get_image_file_path(image_filename)

        keypoints = np.insert(self.part[idx], 2, self.visible[idx], axis=1)[np.newaxis, :, :]

        return image_filename, keypoints, self.visible[idx], self.center[idx], self.scale[idx], self.normalize[idx]

    def split_data(self):
        """
        Split the MPII dataset into training and validation set. Returns index for train and validation imgs.
        Indices for validation images starts after that of train images so that load_image can tell them apart.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: Tuple containing:
                - numpy.ndarray: training data indices with shape (N_train,)
                - numpy.ndarray: validation data indices with shape (N_validation,)
        """
        train = [i for i in range(self.n_train_samples)]
        valid = [i + self.n_train_samples for i in range(self.n_validation_samples)]
        return np.array(train), np.array(valid)


class GenerateHeatmap:
    """
    Generate heatmap for data. Creates a callable object which returns the heatmap based on the input arguments.

    Args:
        output_resolution (int): Output resolution. Default is ``64``.
        num_parts (int): Number of parts. This number depends on the dataset. MPII has ``16`` parts. Refer the ``config.yaml`` file for details under ``["data"]["MPII"]["parts"]`` attribute for more details.

    Examples:
        >>> gen_heatmap = GenerateHeatmap(output_resolution=64, num_parts=16)
        >>> keypoints = np.random.rand(1, 16, 3)        # keypoints for MPII dataset
        >>> heat_map = gen_heatmap(keypoints)           # output shape is (16, 64, 64)
    """

    def __init__(self, output_resolution=64, num_parts=16):
        self.output_resolution = output_resolution
        self.num_parts = num_parts
        sigma = self.output_resolution / 64.
        self.sigma = sigma
        self.gaussian, self.x_center, self.y_center = utils.create_gaussian_kernel(sigma=self.sigma)

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


class MPIIDataset(Dataset):
    """
    MPII dataset handling.

    Args:
        indices (numpy.ndarray): Indices of the dataset.
        mpii_annotation_handle (MPIIAnnotationHandler): Object to handle MPII dataset annotations.
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
    """
    def __init__(self,
                 indices,
                 mpii_annotation_handle,
                 transform=ToTensor(),
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

        self.transform = transform

        self.input_resolution = input_resolution
        self.output_resolution = output_resolution
        self.indices = indices
        self.reference_image_size = reference_image_size
        self.mpii_annotation_handle = mpii_annotation_handle
        self.max_rotation_angle = max_rotation_angle
        self.image_scaling_factor = {"min": image_scale_factor_range[0], "max": image_scale_factor_range[1]}
        self.image_color_jitter_probability = image_color_jitter_probability
        self.image_horizontal_flip_probability = image_horizontal_flip_probability
        self.hue_max_delta = hue_max_delta
        self.saturation_min_delta = saturation_min_delta
        self.brightness_max_delta = brightness_max_delta
        self.contrast_min_delta = contrast_min_delta

        self.generate_heatmap = GenerateHeatmap(self.output_resolution, num_parts)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, heatmaps = self.__load_data__(self.indices[idx % len(self.indices)])

        if self.transform:
            image = self.transform(image)
        heatmaps = tensor(heatmaps)
        return image, heatmaps

    def __load_data__(self, idx):
        """
        Load the image at the given index `idx` of the dataset.

        Args:
            idx (int): index of the data sample.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: Tuple of shape ``(2,)`` containing:
                - numpy.ndarray: Input image of shape ``(H_in, W_in, C_in)``. Mostly as ``H_in=W_in=256`` and channels ``C_in=3``. The input image is normalized to have pixel values between ``[0, 1]``. The content of the numpy array is of type ``numpy.float32``.
                - numpy.ndarray: Output heatmaps containing keypoints on the person's body of shape ``(Parts, H_out, W_out)``. For MPII dataset ``(Parts, H_out, W_out)=(16, 64, 64)``. The content of the numpy array is of ``numpy.float32``.
        """

        image_path, original_keypoints, visibility_flag, c, s, normalize = self.mpii_annotation_handle.get_annotation(idx)

        image = imread(image_path)      # load image with shape ``(H, W, C)``.

        # transform original image to desired input resolution and scale
        image = utils.crop(img=image, center=c, scale=s, resolution=(self.input_resolution, self.input_resolution), rotation=0)

        input_keypoints = self.__transform_original_keypoints_to_input_resolution(original_keypoints, center=c, scale=s)

        height, width = image.shape[0:2]                                    # height and width of input image
        center = np.array((width / 2, height / 2))                          # center of input image
        scale = max(height, width) / self.reference_image_size              # scale input image.

        image, output_keypoints = self.__augment_input_output_with_rotation_and_scale(image, input_keypoints, center, scale)

        if np.random.rand() > self.image_color_jitter_probability:
            image = self.__random_color_jitter(image)

        if np.random.rand() > self.image_horizontal_flip_probability:
            image, input_keypoints, output_keypoints = self.__horizontal_image_flip(image, input_keypoints, output_keypoints)

        # set keypoints to 0 when were not visible initially (so heatmap all 0s)
        for i in range(np.shape(input_keypoints)[1]):
            if original_keypoints[0, i, 0] == 0 and original_keypoints[0, i, 1] == 0:
                output_keypoints[0, i, 0] = 0
                output_keypoints[0, i, 1] = 0
                input_keypoints[0, i, 0] = 0
                input_keypoints[0, i, 1] = 0

        heatmaps = self.generate_heatmap(output_keypoints)      # generate heatmaps on output resolution

        return image.astype(np.float32), heatmaps.astype(np.float32)

    def __transform_original_keypoints_to_input_resolution(self, original_keypoints, center, scale):
        """
        Transform original keypoint to desired input resolution scale.

        Args:
            original_keypoints (numpy.ndarray): Keypoint coordinates with shape ``(1, 16, 3)``.
            center (numpy.ndarray): Center coordinates with shape ``(2,)``.
            scale (float): Scale factor.

        Returns:
            numpy.ndarray: Keypoint coordinates transformed to desired input resolution with shape ``(1, 16, 3)``.

        """
        # Transform pixel locations of keypoints to desired input image resolution
        input_keypoints = original_keypoints.copy()
        for i in range(np.shape(input_keypoints)[1]):
            if input_keypoints[0, i, 0] > 0:
                input_keypoints[0, i, :2] = utils.transform(pt=input_keypoints[0, i, :2], center=center, scale=scale, resolution=(self.input_resolution, self.input_resolution))

        return input_keypoints

    def __horizontal_image_flip(self, image, input_keypoints, output_keypoints):
        """
        Horizontal Flip of the image.

        Args:
            image (numpy.ndarray): Input image of shape ``(H, W, C)``.
            input_keypoints (numpy.ndarray): Keypoint locations in input image with shape ```(1, 16, 3)```.
            output_keypoints (numpy.ndarray): Keypoint locations in output image with shape ```(1, 16, 3)```.

        Returns:
            tuple: Tuple containing following elements in given order:
                - image (numpy.ndarray): Flipped image of same shape as input ``(H, W, C)``.
                - input_keypoints (numpy.ndarray): Flipped location of input image keypoints with shape ``(1, 16, 3)``.
                - output_keypoints (numpy.ndarray): Flipped location of output heatmap keypoints with shape ``(1, 16, 3)``.

        """
        flipped_keypoint_ids = self.mpii_annotation_handle.keypoint_info['flipped_ids']

        image = image[:, ::-1]          # horizontal flip image, WIDTH (a.k.a. X-axis) coordinates of the image are flipped.

        output_keypoints = output_keypoints[:, flipped_keypoint_ids]
        output_keypoints[:, :, 0] = self.output_resolution - output_keypoints[:, :, 0]

        input_keypoints = input_keypoints[:, flipped_keypoint_ids]
        input_keypoints[:, :, 0] = self.input_resolution - input_keypoints[:, :, 0]

        return image, input_keypoints, output_keypoints

    def __augment_input_output_with_rotation_and_scale(self, image, keypoints, center, scale):
        """
        Augment input image and corresponding output target-keypoint-pixel-locations with rotation and scaling factor.

        Args:
            image (numpy.ndarray): Image sample with shape ``(H, W, C)``.
            keypoints (numpy.ndarray): Keypoints in the input image resolution with Shape ``(1, 16, 3)``
            center (numpy.ndarray): Center of image sample with shape ``(2,)``
            scale (float): Original scale factor.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: Tuple of length ``2`` with following elements in the given order:
                - image (numpy.ndarray): Transformed Image after rotation and scaling augmentation.
                - keypoints (numpy.ndarray): Transformed keypoints after rotation and scaling augmentation in the output heatmap.

        Notes:
            - ``Input keypoints`` are in the ``input image reference frame``.
            - ``Output keypoints`` are in the ``output heatmap reference frame``.
        """
        augment_rotation_angle, augment_scale_factor = self.__augment_rotation_and_scale()

        scale *= augment_scale_factor       # scale factor augmentation

        input_transformation_matrix = utils.get_transform(center, scale, (self.input_resolution, self.input_resolution), augment_rotation_angle, self.reference_image_size)[:2]
        image = warpAffine(image, input_transformation_matrix, (self.input_resolution, self.input_resolution)).astype(np.float32) / 255.  # image (data) augmentation

        # transform the output keypoints to be included in heatmaps as per the augmentation
        keypoint_transformation_matrix = utils.get_transform(center, scale, (self.output_resolution, self.output_resolution), augment_rotation_angle, self.reference_image_size)[:2]
        keypoints[:, :, 0:2] = utils.kpt_affine(keypoints[:, :, 0:2], keypoint_transformation_matrix)

        return image, keypoints

    def __augment_rotation_and_scale(self):
        """
        Sample rotation and scaling factor for image augmentation.

        Returns:
            tuple[float, float]: Tuple of length `2` containing rotation angle (in degrees) and scaling factor for augmentation.
        """
        # random rotation angle
        augment_rotation_angle = (np.random.random() * 2 - 1) * self.max_rotation_angle

        # random scale factor
        augment_scale_factor = np.random.random() * (self.image_scaling_factor['max'] - self.image_scaling_factor['min']) + self.image_scaling_factor['min']
        return augment_rotation_angle, augment_scale_factor

    def __random_color_jitter(self, data):
        """
        Color jittering of input image.

        Args:
            data (numpy.ndarray): Image with pixel values scaled between ``[0, 1]`` of shape ``(H, W, C)``.

        Returns:
            numpy.ndarray: Augmented image sample with same as input shape.
        """

        # image from RGB to HSV
        data = cvtColor(data, COLOR_RGB2HSV)

        # hue augmentation
        delta = (np.random.random() * 2 - 1) * self.hue_max_delta
        data[:, :, 0] = np.mod(data[:, :, 0] + (delta * 360 + 360.), 360.)

        # saturation augmentation
        delta_sature = np.random.random() + self.saturation_min_delta
        data[:, :, 1] *= delta_sature
        data[:, :, 1] = np.maximum(np.minimum(data[:, :, 1], 1), 0)

        # image from HSV to RGB
        data = cvtColor(data, COLOR_HSV2RGB)

        # adjust brightness
        delta = (np.random.random() * 2 - 1) * self.brightness_max_delta
        data += delta

        # adjust contrast
        mean = data.mean(axis=2, keepdims=True)
        data = (data - mean) * (np.random.random() + self.contrast_min_delta) + mean
        data = np.minimum(np.maximum(data, 0), 1)
        return data


# def init(config):
#     """
#     Initialize data loader
#
#     Args:
#         config (ConfigYamlParserMPII): Configuration parsed using ``config.yaml`` file for MPII dataset.
#
#     Returns:
#         types.LambdaType:
#     """
#
#     batch_size = int(config.NN_TRAINING_PARAMS['batch_size'])
#     input_resolution = int(config.NN_TRAINING_PARAMS['input_resolution'])
#     output_resolution = int(config.NN_TRAINING_PARAMS['output_resolution'])
#     num_workers = int(config.NN_TRAINING_PARAMS['num_workers'])
#
#     num_parts = int(config.PARTS['max_count'])
#
#     mpii_annotation_handle = MPIIAnnotationHandler(
#         training_annotation_file=config.TRAINING_ANNOTATION_FILE,
#         validation_annotation_file=config.VALIDATION_ANNOTATION_FILE,
#         image_dir=config.IMAGE_DIR,
#         keypoint_info=config.PARTS)
#
#     training_data_indices, validation_data_indices = mpii_annotation_handle.split_data()
#
#     __data_aug = config.NN_TRAINING_PARAMS['data_augmentation']
#     __img_sf = __data_aug['image_scale_factor']
#     image_scale_factor_range = (__img_sf['min'], __img_sf['max'])
#     dataset = {key: MPIIDataset(
#                         input_resolution=input_resolution,
#                         output_resolution=output_resolution,
#                         num_parts=num_parts,
#                         indices=indices,
#                         mpii_annotation_handle=mpii_annotation_handle,
#                         reference_image_size=config.REFERENCE_IMAGE_SIZE,
#                         max_rotation_angle=__data_aug['rotation_angle_max'],
#                         image_scale_factor_range=image_scale_factor_range,
#                         image_color_jitter_probability=__data_aug['image_color_jitter_probability'],
#                         hue_max_delta=__data_aug['hue_max_delta'],
#                         saturation_min_delta=__data_aug['saturation_min_delta'],
#                         brightness_max_delta=__data_aug['brightness_max_delta'],
#                         contrast_min_delta=__data_aug['contrast_min_delta']
#                     ) for key, indices in zip(['train', 'valid'], [training_data_indices, validation_data_indices])}
#
#     loaders = {}
#     for key in dataset:
#         loaders[key] = torch.utils.data.DataLoader(dataset[key], batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
#
#     def gen(phase):
#         """
#         Args:
#             phase (str): Phase can take one value out of ('train', 'valid') for training and validation respectively.
#
#         Yields:
#             dict: Dictionary containing keys 'imgs' and 'heatmaps' corresponding to images and heatmaps returned in each batch.
#         """
#         iters = config.NN_TRAINING_PARAMS['{}_iterations'.format(phase)]
#         loader = loaders[phase].__iter__()
#         for i in range(iters):
#             try:
#                 imgs, heatmaps = next(loader)
#             except StopIteration:   # to avoid no data provided by dataloader
#                 loader = loaders[phase].__iter__()
#                 imgs, heatmaps = next(loader)
#             yield {
#                 'imgs': imgs,               # cropped and augmented
#                 'heatmaps': heatmaps,       # based on keypoints. 0 if not in img for joint
#             }
#     return lambda key: gen(key)
