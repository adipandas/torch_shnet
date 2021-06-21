import numpy as np
import cv2
import torch


REFERENCE_PIXEL_SIZE = 200
"""
int: Reference pixel size of Person's image. MPII dataset stores the scale with reference to `200 pixel` size.

References:
    * MPII website: http://human-pose.mpi-inf.mpg.de/#download
"""


def make_input(x, requires_grad=False, need_cuda=True):
    """
    Make the input for network.

    Args:
        x (torch.tensor or numpy.ndarray):
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
        x (torch.tensor or list[torch.tensor] or tuple[torch.tensor]): tensor (or list of tensors) from torch module

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
    return np.array([cv2.resize(imgs[i], resolution) for i in range(imgs.shape[0])])


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


def get_transform(center, scale, resolution, rotation=0):
    """
    Generate transformation matrix.

    Args:
        center (numpy.ndarray or tuple or list): Array of shape (2,) containing x, y coordinates of the center.
        scale (float): person scale w.r.t. 200 px height.
        resolution (tuple or list or numpy.ndarray): Desired resoultion as tuple ``(Height, Width)`` of length `2`.
        rotation (float): Angle of rotation in degrees. Default is `0`.

    Returns:
        numpy.ndarray: Transformation matrix of shape (3, 3).
    """

    h = REFERENCE_PIXEL_SIZE * scale                        # current resolution
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

    return cv2.resize(new_img, resolution)
