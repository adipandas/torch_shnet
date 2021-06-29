import os.path as osp
import time
import numpy as np
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

    Args:
        training_annotation_file (str): Path to annotations (a.k.a. labels) file for training set in ``.h5`` format.
        validation_annotation_file (str): Path to annotations (a.k.a. labels) file for validation set in ``.h5`` format.
        image_dir (str): Path to directory containing the MPII dataset images.

    References:
        * https://dbcollection.readthedocs.io/en/latest/datasets/mpii_pose.html
    """

    def __init__(self, training_annotation_file, validation_annotation_file, image_dir):
        print('Loading annotation data...')
        tic = time.time()

        self.image_dir = image_dir

        (self.center, self.scale, self.part, self.visible, self.normalize, self.image_filename, self.n_train_samples, self.n_validation_samples) = utils.load_MPII_training_and_validation_annotation_file(training_annotation_file, validation_annotation_file)

        print('Done! (t={:0.2f}s)'.format(time.time() - tic))

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
            image_filename = osp.join(self.image_dir, image_filename)

        keypoints = np.insert(self.part[idx], 2, self.visible[idx], axis=1)[np.newaxis, :, :]

        return image_filename, keypoints, self.visible[idx], self.center[idx], self.scale[idx], self.normalize[idx]
