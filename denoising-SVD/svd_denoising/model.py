import torch
import torch.nn as nn


def conv3x3(in_channels: int, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)


class UnetN2N(nn.Module):
    """Noise2Noise U-Net adapted for two-channel complex tensors."""

    def __init__(self, in_channels: int = 2, out_channels: int = 2) -> None:
        super().__init__()

        self.enc_conv0 = conv3x3(in_channels, 48)
        self.enc_relu0 = nn.LeakyReLU(0.1)
        self.enc_conv1 = conv3x3(48, 48)
        self.enc_bn1 = nn.BatchNorm2d(48)
        self.enc_relu1 = nn.LeakyReLU(0.1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc_conv2 = conv3x3(48, 48)
        self.enc_bn2 = nn.BatchNorm2d(48)
        self.enc_relu2 = nn.LeakyReLU(0.1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc_conv3 = conv3x3(48, 48)
        self.enc_bn3 = nn.BatchNorm2d(48)
        self.enc_relu3 = nn.LeakyReLU(0.1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc_conv4 = conv3x3(48, 48)
        self.enc_bn4 = nn.BatchNorm2d(48)
        self.enc_relu4 = nn.LeakyReLU(0.1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc_conv5 = conv3x3(48, 48)
        self.enc_bn5 = nn.BatchNorm2d(48)
        self.enc_relu5 = nn.LeakyReLU(0.1)
        self.pool5 = nn.MaxPool2d(kernel_size=2)

        self.enc_conv6 = conv3x3(48, 48)
        self.enc_bn6 = nn.BatchNorm2d(48)
        self.enc_relu6 = nn.LeakyReLU(0.1)
        self.upsample5 = nn.Upsample(scale_factor=2, mode="nearest")

        self.dec_conv5a = conv3x3(96, 96)
        self.dec_bn5a = nn.BatchNorm2d(96)
        self.dec_relu5a = nn.LeakyReLU(0.1)
        self.dec_conv5b = conv3x3(96, 96)
        self.dec_bn5b = nn.BatchNorm2d(96)
        self.dec_relu5b = nn.LeakyReLU(0.1)
        self.upsample4 = nn.Upsample(scale_factor=2, mode="nearest")

        self.dec_conv4a = conv3x3(144, 96)
        self.dec_bn4a = nn.BatchNorm2d(96)
        self.dec_relu4a = nn.LeakyReLU(0.1)
        self.dec_conv4b = conv3x3(96, 96)
        self.dec_bn4b = nn.BatchNorm2d(96)
        self.dec_relu4b = nn.LeakyReLU(0.1)
        self.upsample3 = nn.Upsample(scale_factor=2, mode="nearest")

        self.dec_conv3a = conv3x3(144, 96)
        self.dec_bn3a = nn.BatchNorm2d(96)
        self.dec_relu3a = nn.LeakyReLU(0.1)
        self.dec_conv3b = conv3x3(96, 96)
        self.dec_bn3b = nn.BatchNorm2d(96)
        self.dec_relu3b = nn.LeakyReLU(0.1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")

        self.dec_conv2a = conv3x3(144, 96)
        self.dec_bn2a = nn.BatchNorm2d(96)
        self.dec_relu2a = nn.LeakyReLU(0.1)
        self.dec_conv2b = conv3x3(96, 96)
        self.dec_bn2b = nn.BatchNorm2d(96)
        self.dec_relu2b = nn.LeakyReLU(0.1)
        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")

        self.dec_conv1a = conv3x3(96 + in_channels, 64)
        self.dec_bn1a = nn.BatchNorm2d(64)
        self.dec_relu1a = nn.LeakyReLU(0.1)
        self.dec_conv1b = conv3x3(64, 32)
        self.dec_bn1b = nn.BatchNorm2d(32)
        self.dec_relu1b = nn.LeakyReLU(0.1)
        self.dec_conv1c = conv3x3(32, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.enc_relu0(self.enc_conv0(x))
        e1 = self.pool1(self.enc_relu1(self.enc_bn1(self.enc_conv1(e0))))
        e2 = self.pool2(self.enc_relu2(self.enc_bn2(self.enc_conv2(e1))))
        e3 = self.pool3(self.enc_relu3(self.enc_bn3(self.enc_conv3(e2))))
        e4 = self.pool4(self.enc_relu4(self.enc_bn4(self.enc_conv4(e3))))
        e5 = self.pool5(self.enc_relu5(self.enc_bn5(self.enc_conv5(e4))))
        d5 = self.upsample5(self.enc_relu6(self.enc_bn6(self.enc_conv6(e5))))
        d4 = self.upsample4(self.dec_relu5b(self.dec_bn5b(self.dec_conv5b(self.dec_relu5a(self.dec_bn5a(self.dec_conv5a(torch.cat((d5, e4), 1))))))))
        d3 = self.upsample3(self.dec_relu4b(self.dec_bn4b(self.dec_conv4b(self.dec_relu4a(self.dec_bn4a(self.dec_conv4a(torch.cat((d4, e3), 1))))))))
        d2 = self.upsample2(self.dec_relu3b(self.dec_bn3b(self.dec_conv3b(self.dec_relu3a(self.dec_bn3a(self.dec_conv3a(torch.cat((d3, e2), 1))))))))
        d1 = self.upsample1(self.dec_relu2b(self.dec_bn2b(self.dec_conv2b(self.dec_relu2a(self.dec_bn2a(self.dec_conv2a(torch.cat((d2, e1), 1))))))))
        return self.dec_conv1c(self.dec_relu1b(self.dec_bn1b(self.dec_conv1b(self.dec_relu1a(self.dec_bn1a(self.dec_conv1a(torch.cat((d1, x), 1))))))))

    @property
    def model_size(self) -> tuple[int, int]:
        n_params = sum(param.numel() for param in self.parameters())
        n_conv_layers = sum(1 for module in self.modules() if isinstance(module, nn.Conv2d))
        return n_params, n_conv_layers
