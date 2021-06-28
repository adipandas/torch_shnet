import numpy as np
from cv2 import resize as cv2_resize
import torch
from torch import nn
import h5py


def _is_array_like(obj_):
    """
    Check if the input object is like an array or not.

    Args:
        obj_ (object): Input object.

    Returns:
        bool: `True` if the input object has ``__iter__`` and ``__len__`` attributes.
    """
    return hasattr(obj_, '__iter__') and hasattr(obj_, '__len__')


def make_input(x, requires_grad=False, need_cuda=True):
    """
    Make the input for network. Converts input to pytorch tensor.

    Args:
        x (torch.Tensor or numpy.ndarray):
        requires_grad (bool): If `True`, track the gradient of torch variable. Default is ``False``.
        need_cuda (bool): If ``True``, add the created torch tensor to the GPU device. Default is ``True``.

    Returns:
        torch.tensor:
    """
    inp = torch.autograd.Variable(x, requires_grad=requires_grad)
    if need_cuda:
        inp = inp.cuda()
    return inp


def make_output(x):
    """
    Convert the data from torch tensors to numpy array format.

    Args:
        x (torch.Tensor or list[torch.Tensor] or tuple[torch.Tensor]): tensor (or list of tensors) from torch module

    Returns:
        numpy.ndarray or list[numpy.ndarray]: Numpy tensor (or list of numpy tensors).
    """
    if not (type(x) is list):
        return x.cpu().data.numpy()
    else:
        return [make_output(i) for i in x]


def resize(imgs, resolution):
    """
    Args:
        imgs (numpy.ndarray): Stack of images as (N, H, W, C) or (N, H, W)
        resolution (tuple): Tuple of length 2 containing (width, height)=(W_new, H_new)

    Returns:
        numpy.ndarray: Output of image or stack of image with shape (N, H_new, W_new, C) or (N, H_new, W_new)

    Notes:
        * Although `resolution` argument for this function takes ``desired width`` as the first element followed by ``desired height``,
          opencv treats images as ``(H, W, C)`` or ``(H, W)`` where ``H=image_height`` comes first followed by ``W=image_width``.
    """
    return np.array([cv2_resize(imgs[i], resolution) for i in range(imgs.shape[0])])


def inv_mat(mat):
    """
    Calculate matrix inverse (matrix pseudo inverse is evaluated).

    Args:
        mat (numpy.ndarray): Matrix of shape (2, 3).

    Returns:
        numpy.ndarray: Matrix of shape (2, 3).

    Notes:
        * The input matrix is suppose to be a transformation matrix of shape (2, 3) that is internally converted to
         square matrix (3, 3) before inverse calculation. The last row of the (3, 3) matrix is dropped and (2, 3) is returned.
    """

    assert len(mat) == 2 and len(mat[0]) == 3, "Input shape must be (2, 3)"
    ans = np.linalg.pinv(np.array(mat).tolist() + [[0, 0, 1]])
    return ans[:2]


def kpt_affine(kpt, mat):
    """
    Keypoint affine transformation.

    Args:
        kpt (numpy.ndarray or list or tuple): Keypoint coordinates of shape ``(N, 2)`` or ``(2*N,)``.
        mat (numpy.ndarray): Transformation matrix of shape ``(2, 3)``.

    Returns:
        numpy.ndarray: Transformed keypoint array of shape ``(N, 2)`` or ``(2*N,)``.
    """
    kpt = np.array(kpt)
    shape = kpt.shape  # original input shape
    kpt = kpt.reshape(-1, 2)
    ones_ = np.ones((kpt.shape[0], 1))
    kpt = np.concatenate((kpt, ones_), axis=1)  # shape (N, 3) with each row as [x, y, 1].
    new_kpt = np.dot(kpt, mat.T)
    new_kpt = new_kpt.reshape(shape)
    return new_kpt


def get_transform(center, scale, resolution, rotation=0, reference_image_size=200):
    """
    Generate transformation matrix.

    Args:
        center (numpy.ndarray or tuple or list): Array of shape (2,) containing x, y coordinates of the center.
        scale (float): person scale w.r.t. 200 px height.
        resolution (tuple or list or numpy.ndarray): Desired resoultion as tuple ``(Height, Width)`` of length `2`.
        rotation (float): Angle of rotation in degrees. Default is `0`.
        reference_image_size (int): Scale of the image used as refernce for scaling original data.

    Returns:
        numpy.ndarray: Transformation matrix of shape (3, 3).
    """

    h = reference_image_size * scale                        # current resolution
    xc, yc = float(center[0]), float(center[1])             # center as per current resolution

    X, Y = float(resolution[1]), float(resolution[0])       # desired resolution
    xc_des, yc_des = X*0.5, Y*0.5                           # desired center

    t = np.zeros((3, 3))
    t[0, 0] = X/h
    t[1, 1] = Y/h
    t[0, 2] = - xc * X / h + xc_des
    t[1, 2] = - yc * Y / h + yc_des
    t[2, 2] = 1

    if not rotation == 0:
        rotation = - rotation                     # To match direction of rotation from cropping
        rotation = rotation * np.pi / 180.0       # convert to radians
        sn, cs = np.sin(rotation), np.cos(rotation)

        rot_mat = np.zeros((3, 3))
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1

        # Need to rotate around center: Move image center from midpoint to origin (0, 0)
        translation_mat = np.eye(3)
        translation_mat[0, 2] = - xc_des
        translation_mat[1, 2] = - yc_des

        # Move image center from origin (0, 0) to midpoint
        translation_mat_inv = translation_mat.copy()
        translation_mat_inv[:2, 2] *= -1

        t = np.dot(translation_mat, t)  # move image center from midpoint to origin (0, 0)
        t = np.dot(rot_mat, t)          # rotate image about origin by `rotation` angle along z-axis
        t = np.dot(translation_mat_inv, t)   # move image center from origin (0, 0) to midpoint

    return t


def transform(pt, center, scale, resolution, invert=False, rotation=0):
    """
    Transform pixel location to different reference

    Args:
        pt (numpy.ndarray or tuple or list): Point denoting pixel location of shape ``(2,)`` as ``(x, y)``.
        center (numpy.ndarray or tuple or list): Array of shape ``(2,)`` containing ``(x, y)`` coordinates of the center.
        scale (float): person scale w.r.t. ``200 px`` height.
        resolution (tuple or list or numpy.ndarray): Desired resoultion as tuple ``(Height, Width)`` of length `2`.
        invert (bool): If ``True``, calculate inverse of transformation matrix. Default is ``False``.
        rotation (float): Angle of rotation in ``degrees``. Default is ``0``.

    Returns:
        numpy.ndarray: Transformed pixel locations as `int` array of shape (2,).
    """
    t = get_transform(center, scale, resolution, rotation=rotation)
    if invert:
        t = np.linalg.inv(t)

    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int)


def crop(img, center, scale, resolution, rotation=0):
    """
    Crop input image.

    Args:
        img (numpy.ndarray): Input image of shape ``(H_in, W_in, C)`` or ``(H_in, W_in)``.
        center (numpy.ndarray or tuple or list): Array of shape ``(2,)`` containing ``(x, y)`` coordinates of the center.
        scale (float): person scale w.r.t. ``200 px`` height.
        resolution (tuple or list or numpy.ndarray): Desired resoultion as tuple containing ``(H, W)`` of length `2`.
        rotation (float): Angle of rotation in ``degrees``. Default is ``0``.

    Returns:
        numpy.ndarray: New image of shape ``(H, W, C)`` or ``(H, W)``.

    Notes:
        - ``H`` is height of image.
        - ``W`` is width of image.
        - ``C`` is number of channels in image.
    """

    ul = np.array(transform([0, 0], center, scale, resolution, invert=True, rotation=rotation))       # Upper left point
    br = np.array(transform(resolution, center, scale, resolution, invert=True, rotation=rotation))   # Bottom right point

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]

    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])

    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    return cv2_resize(new_img, resolution)


class HeatmapParser:
    def __init__(self):
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def nms(self, det):
        """
        Non-maximal suppression of detected heatmap.

        Args:
            det (torch.Tensor): Detections as heatmaps with shape (N, C, H, W).

        Returns:
            torch.Tensor: Non-maximal supression of input with shape (N, C, H, W).

        Notes:
            N - Batch Size
            C - Channel - this is 16 for MPII dataset
            H - Height  - (output of hourglass network = 64)
            W - Width   - (output of hourglass network = 64)
        """
        maxm = self.pool(det)
        maxm = torch.eq(maxm, det).float()   # element wise equality
        det = det * maxm
        return det

    def calc(self, det):
        """
        Predict the keypoint pixel locations and the pixel value from the detection heatmap ``det`` tensor.

        Args:
            det (torch.Tensor): Detections as heatmaps with shape (N, C, H, W).

        Returns:
            dict[str, numpy.ndarray]: Dictionary containing following keys:
                - "loc_k": with corresponding value as numpy.ndarray of indices with shape ``(N, C, 1, 2)``.
                - "val_k": With corresponding value as numpy.ndarray of value at the ``loc_k`` index with shape ``(N, C, 1)``.

        Notes:
            N - Batch Size
            C - Channel - this is 16 for MPII dataset
            H - Height  - (output of hourglass network = 64)
            W - Width   - (output of hourglass network = 64)

        """

        with torch.no_grad():
            det = torch.autograd.Variable(torch.Tensor(det))

        det = self.nms(det)

        h = det.size()[2]
        w = det.size()[3]

        det = det.view(det.size()[0], det.size()[1], -1)

        val_k, ind = det.topk(1, dim=2)

        x = ind % w
        y = (ind / h).long()

        ind_k = torch.stack((x, y), dim=3)

        ans = {'loc_k': ind_k, 'val_k': val_k}
        return {key: ans[key].cpu().data.numpy() for key in ans}

    def adjust(self, keypoint_list, detection):
        """
        Adjust detected keypoint locations as per the intensity of its neighboring pixel in the detection heatmap.

        Args:
            keypoint_list (list[numpy.ndarray]): Each element of the list is Keypoint pixel-locations and pixel-values in the detection heatmap with shape ``(1, C, 3)`` where ``C`` is number of keypoints. ``C=16`` for MPII dataset.
            detection (torch.Tensor): Detection heatmap of shape ``(N, C, H, W)``. ``(N, C, H, W)=(N, 16, 64, 64)`` based on network output specifications. Refer ``config.yaml``. Batch size ``N=1`` for testing/inference.

        Returns:
            list[numpy.ndarray]: List of length `1`. The element of the list is ``numpy.ndarray`` with shape ``(1, C, 3)``.

        """
        for batch_id, people in enumerate(keypoint_list):
            for people_id, i in enumerate(people):
                for joint_id, joint in enumerate(i):
                    if joint[2] > 0:                # predicted keypoint intensity is greater than zero
                        w, h = joint[0:2]           # joint coordinates along width (w or x) and height (h or y) of detection image
                        hh, ww = int(h), int(w)

                        tmp = detection[0][joint_id]   # channel corresponding to joint_id in the detection heatmap.

                        if tmp[hh, min(ww + 1, tmp.shape[1] - 1)] > tmp[hh, max(ww - 1, 0)]:  # right neighbor intensity > left neighbor intensity
                            w += 0.25
                        else:
                            w -= 0.25

                        if tmp[min(hh + 1, tmp.shape[0] - 1), ww] > tmp[max(0, hh - 1), ww]:  # down neighbor intensity > up neighbor intensity
                            h += 0.25
                        else:
                            h -= 0.25
                        keypoint_list[0][0, joint_id, 0:2] = (w + 0.5, h + 0.5)
        return keypoint_list

    @staticmethod
    def match_format(dic):
        """
        Convert the input dictionary of the matched keypoint to a list of numpy.ndarray.

        Args:
            dic (dict[str, numpy.ndarray]): Dictionary containing following keys:
                    - "loc_k": with corresponding value as numpy.ndarray of indices with shape ``(1, C, 1, 2)``.
                    - "val_k": With corresponding value as numpy.ndarray of value at the ``loc_k`` index with shape ``(1, C, 1)``.

        Returns:
            list[numpy.ndarray]: List of length `1`. The element of the list is ``numpy.ndarray`` with shape ``(1, C, 3)``. Each row of the numpy.ndarray is (x, y, pixel-intensity) of detected keypoint. For MPII dataset ``C=16``.

        """
        loc = dic['loc_k'][0, :, 0, :]  # shape    (C, 2)=(16, 2)
        val = dic['val_k'][0, :, :]  # shape    (C, 1)=(16, 1)
        ans = np.hstack((loc, val))  # shape    (C, 3)=(16, 3)
        ans = np.expand_dims(ans, axis=0)  # shape (1, C, 3)=(1, 16, 3)
        ret = [ans]
        return ret

    def parse(self, det, adjust=True):
        """
        Parse the input detection heatmap.

        Args:
            det (torch.Tensor): Detection heatmap from network output with shape ``(1, C, H, W)=(1, 16, 64, 64)``
            adjust (bool): If ``True``, adjust detected keypoint locations as per the intensity of neighboring pixel on heatmap. Default is ``True``.

        Returns:
            list[numpy.ndarray]: List of length `1`. The element of the list is ``numpy.ndarray`` with shape ``(1, C, 3)``.
        """
        kp_dict = self.calc(det)
        kp_list = HeatmapParser.match_format(kp_dict)
        if adjust:
            kp_list = self.adjust(kp_list, det)
        return kp_list


def create_gaussian_kernel(sigma=1.0, size=None):
    """
    Method to create a square 2D gaussian kernel.

    Args:
        sigma (float): Standard deviation of gaussian kernel.
        size (int): Size of gaussian kernel. Default is ``None``. If ``None``, the default kernel_size is ``9``. Input value to be an odd number.

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


def load_MPII_annotation_file(annotation_file):
    """
    Load the annotation file for MPII dataset.

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


def load_MPII_training_and_validation_annotation_file(training_annotation_file, validation_annotation_file):
    """
    Load training and validation annotation files for MPII dataset.

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
    tcenter, tscale, tpart, tvisible, tnormalize, tdata_length, tfilename = load_MPII_annotation_file(training_annotation_file)
    vcenter, vscale, vpart, vvisible, vnormalize, vdata_length, vfilename = load_MPII_annotation_file(validation_annotation_file)

    center = np.append(tcenter, vcenter, axis=0)
    scale = np.append(tscale, vscale)
    part = np.append(tpart, vpart, axis=0)
    visible = np.append(tvisible, vvisible, axis=0)
    normalize = np.append(tnormalize, vnormalize)
    filename = tfilename + vfilename

    training_data_length = tdata_length
    validation_data_length = vdata_length

    return center, scale, part, visible, normalize, filename, training_data_length, validation_data_length
