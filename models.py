import torch
from torch import nn


class Conv2DCustom(nn.Module):
    """
    Custom convolutional layer that can also apply batch normalization and ReLU activation after convolution.

    Args:
        in_channels (int): Input channels
        out_channels (int): Output channels
        kernel_size (int): Kernel size of convolutional kernel. Default is `3`.
        stride (int): Stride in convolution. Default is `1`.
        batch_normalization (bool): If `True`, use batch normalization operation after convolution. Default is `False`.
        relu_activation (bool): If `True`, use ReLU activation after convolution. Default is `True`.

    Shape:
        - input: (N, C_in, H_in, W_in)
        - output: (N, C_out, H_out, W_out)
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, batch_normalization=False,
                 relu_activation=True):
        super(Conv2DCustom, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=True)

        if relu_activation:
            self.relu = nn.ReLU()
        else:
            self.relu = None

        if batch_normalization:
            self.bn = nn.BatchNorm2d(num_features=out_channels)
        else:
            self.bn = None

    def forward(self, x):
        """
        Forward pass. For more information on input and output tensor dimensions check https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html.

        Args:
            x (torch.Tensor): Shape (N, C_in, H_in, W_in)

        Return:
            torch.Tensor: Shape (N, C_out, H_out, W_out)
        """

        assert x.size()[
                   1] == self.in_channels, f"Check number of channels in input batch.  {x.size()[1]} != {self.in_channels}."

        x = self.conv(x)

        if self.bn is not None:
            x = self.bn(x)

        if self.relu is not None:
            x = self.relu(x)

        return x


class ResidualBlock2D(nn.Module):
    """
    Residual Block diagram.

    Args:
        in_channels (int): input number of channels
        out_channels (int): output number of channels

    Shape:
        - input: (N, C_in, H, W)
        - output: (N, C_out, H, W)
    """

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock2D, self).__init__()

        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm2d(num_features=in_channels)
        self.conv1 = Conv2DCustom(in_channels=in_channels, out_channels=int(out_channels / 2), kernel_size=1,
                                  relu_activation=False)

        self.bn2 = nn.BatchNorm2d(num_features=int(out_channels / 2))
        self.conv2 = Conv2DCustom(in_channels=int(out_channels / 2), out_channels=int(out_channels / 2), kernel_size=3,
                                  relu_activation=False)

        self.bn3 = nn.BatchNorm2d(num_features=int(out_channels / 2))
        self.conv3 = Conv2DCustom(in_channels=int(out_channels / 2), out_channels=out_channels, kernel_size=1,
                                  relu_activation=False)

        if in_channels == out_channels:
            self.skip_layer = None
        else:
            self.skip_layer = Conv2DCustom(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                           relu_activation=False)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Shape (N, C_in, H, W)

        Return:
            torch.Tensor: Shape (N, C_out, H, W)
        """
        if self.skip_layer is not None:
            residual = self.skip_layer(x)
        else:
            residual = x

        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.conv3(self.relu(self.bn3(out)))
        out += residual
        return out


class HourGlassRecursive2D(nn.Module):
    """
    2D Recursive Hour-Glass network module.

    Args:
        n_recursion (int): Number of recursions.
        n_channels (int): Channel size.
        channel_increase (int): Channel increase count.

    Shape:
        - input: (N, C, H, W)
        - output: (N, C, H, W)
    """

    def __init__(self, n_recursion, n_channels, channel_increase=0):
        super(HourGlassRecursive2D, self).__init__()

        m_channels = n_channels + channel_increase

        self.recursion_step = n_recursion

        self.skip = ResidualBlock2D(in_channels=n_channels, out_channels=n_channels)
        self.layer1 = ResidualBlock2D(in_channels=n_channels, out_channels=m_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        if self.recursion_step > 1:
            self.layer2 = HourGlassRecursive2D(n_recursion=n_recursion - 1, n_channels=m_channels)
        else:
            self.layer2 = ResidualBlock2D(in_channels=m_channels, out_channels=m_channels)

        self.layer3 = ResidualBlock2D(in_channels=m_channels, out_channels=n_channels)

        self.upsampler = nn.Upsample(scale_factor=2)

    def forward(self, x):
        skip_out = self.skip(x)
        out = self.pool(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.upsampler(out)
        return out + skip_out


class HeatMapLoss(nn.Module):
    """
    Heat map loss function

    Shape:
        - input:
             - prediction: (N, C, H, W)
             - ground_truth: (N, C, H, W)
        - output: (N,)
    """
    def __init__(self):
        super(HeatMapLoss, self).__init__()

    def forward(self, prediction, ground_truth):
        """
        Args:
            prediction (torch.Tensor): Shape (N, C, H, W)
            ground_truth (torch.Tensor): Shape (N, C, H, W)

        Return:
            torch.Tensor: Loss with shape (N,)

        Notes:
            N: Batch-Size
            C: Channel
            H: Height
            W: Width
        """
        loss = (prediction - ground_truth)*(prediction - ground_truth)
        loss = loss.mean(dim=3).mean(dim=2).mean(dim=1)
        return loss


class PoseNet(nn.Module):
    """
    Stacked HourGlass network.

    Args:
        n_hourglass (int): Number of hourglass modules in the network stacked on one another
        in_channels (int): Number of channels in input.
        out_channels (int): Number of channels in output.
        channel_increase (int): Number of channels to increase in hourglass module. Default is `0`.

    Shape:
        - input: (N, C_in, H, W)
        - output: (N, n_hourglass, C_out, (H+1)/4, (W+1)/4)
    """

    def __init__(self, n_hourglass, in_channels, out_channels, channel_increase=0):
        super(PoseNet, self).__init__()

        self.n_hourglass = n_hourglass

        self.pre_process_backbone = nn.Sequential(
            Conv2DCustom(in_channels=3, out_channels=64, kernel_size=7, stride=2, batch_normalization=True,
                         relu_activation=True),
            ResidualBlock2D(in_channels=64, out_channels=128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock2D(in_channels=128, out_channels=128),
            ResidualBlock2D(in_channels=128, out_channels=in_channels)
        )

        self.hourglass_stack = nn.ModuleList(
            [nn.Sequential(
                HourGlassRecursive2D(n_recursion=4, n_channels=in_channels, channel_increase=channel_increase)) for _ in
             range(n_hourglass)]
        )

        self.features = nn.ModuleList([
            nn.Sequential(
                ResidualBlock2D(in_channels=in_channels, out_channels=in_channels),
                Conv2DCustom(in_channels=in_channels, out_channels=in_channels, kernel_size=1, batch_normalization=True,
                             relu_activation=True)
            ) for _ in range(n_hourglass)
        ])

        self.predictions = nn.ModuleList([Conv2DCustom(in_channels=in_channels, out_channels=out_channels,
                                                       kernel_size=1, batch_normalization=False, relu_activation=False)
                                          for _ in range(n_hourglass)])

        self.merge_features = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0, bias=True)
            for _ in range(n_hourglass - 1)
        ])

        self.merge_predictions = nn.ModuleList([
            nn.Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0, bias=True)
            for _ in range(n_hourglass - 1)
        ])

        self.heatmap_loss = HeatMapLoss()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Shape (N, C_in, H, W).

        Return:
            torch.Tensor: Shape (N, n_hourglass, C_out, (H+1)/4, (W+1)/4)

        Notes:
            N: Batch size
            C: Number of channels
            H: Input height
            W: Input width
        """

        x = self.pre_process_backbone(x)

        predictions_stack = []
        for i in range(self.n_hourglass):
            hg_out = self.hourglass_stack[i](x)
            feature_out = self.features[i](hg_out)
            prediction_out = self.predictions[i](feature_out)

            predictions_stack.append(prediction_out)

            if i < (self.n_hourglass - 1):
                x = x + self.merge_features[i](feature_out) + self.merge_predictions[i](prediction_out)

        return torch.stack(predictions_stack, dim=1)

    def calc_loss(self, prediction, ground_truth):
        """
        Evaluate loss given the prediction and groundtruth output values.

        Args:
            prediction (torch.Tensor): Shape ``(N, n_hourglass, C, (H+1)/4, (W+1)/4)``.
            ground_truth (torch.Tensor): Shape ``(N, C, (H+1)/4, (W+1)/4)``.

        Return:
            torch.Tensor: Shape ``(N, n_hourglass)``.

        Notes:
            - ``(H, W)`` is the input image size used for getting the prediction of shape ``((H+1)/4, (W+1)/4)``.

        """
        loss_ = []
        for i in range(self.n_hourglass):
            loss = self.heatmap_loss(prediction=prediction[:, i], ground_truth=ground_truth)
            loss_.append(loss)

        loss_ = torch.stack(loss_, dim=1)
        return loss_
