import os.path as osp
import time
import numpy as np
import utils.helpers as utils


class MPIIAnnotationHandler:
    """
    Class for MPII Dataset Annotation handling. Here only training and validation annotations are handled. Creates iterator object.
    Returns data from h5 file for train or val set corresponding to given index ``idx``.

    TODO: Handle testing annotations in this class. (MAY BE...)

    Args:
        training_annotation_file (str): Path to annotations (a.k.a. labels) file for training set in ``.h5`` format.
        validation_annotation_file (str): Path to annotations (a.k.a. labels) file for validation set in ``.h5`` format.
        image_dir (str): Path to directory containing the MPII dataset images. Default is ``None``.

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

    Notes:
        * The ``part`` attribute is an array of human-pose keypoint coordinates in the image.
        * Returns from ``__getitem__`` method:
            - image_filename (``str``): Path to image file if ``image_dir`` attribute is provided, else image file name.
            - keypoints (``numpy.ndarray``): Keypoints corresponding to index ``idx`` with shape ``(1, 16, 3)`` with each row of idx ``0`` containing ``(x, y, visibility_flag)``.
            - center_coordinates (``numpy.ndarray``): Array containing ``(x, y)`` pixel coordinates of center of human. Shape ``(2,)``.
            - scale_factor (``float``): Scale factor with reference to reference image size. Scale factor for MPII dataset images is calculated w.r.t. ``200`` px.

    Examples:
        >>> training_annotation_file = "training_annotation_file/path"
        >>> validation_annotation_file = "validation_annotation_file/path"
        >>> image_dir = "image_dir/path"
        >>> m = MPIIAnnotationHandler(training_annotation_file, validation_annotation_file, image_dir)
        >>> total_data_length = len(m)   # length of samples in the data
        >>> idx= 1 # index of sample in the dataset
        >>> image_filename, keypoints, center, scale, = m[idx]

    References:
        * https://dbcollection.readthedocs.io/en/latest/datasets/mpii_pose.html
    """

    def __init__(self, training_annotation_file, validation_annotation_file, image_dir=None):
        print('Loading annotation data...')
        tic = time.time()

        self.image_dir = image_dir

        (self.center, self.scale, self.part, self.visible, self.normalize, self.image_filename, self.n_train_samples, self.n_validation_samples) = utils.load_MPII_training_and_validation_annotation_file(training_annotation_file, validation_annotation_file)

        print('Done! (t={:0.2f}s)'.format(time.time() - tic))

    def __len__(self):
        return self.n_train_samples + self.n_validation_samples

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

    def __getitem__(self, idx):
        image_filename = self.image_filename[idx]
        if self.image_dir is not None:
            image_filename = osp.join(self.image_dir, image_filename)
        keypoints = np.insert(self.part[idx], 2, self.visible[idx], axis=1)[np.newaxis, :, :]
        return image_filename, keypoints, self.center[idx], self.scale[idx]
