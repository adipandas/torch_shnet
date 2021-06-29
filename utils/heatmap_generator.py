import numpy as np
import utils.helpers as utils


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
