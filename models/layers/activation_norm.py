import torch
import torch.nn as nn

from models.utils import calc_mean_std


class ToStyleHead(nn.Module):
    def __init__(self, input_dim: int = 512, output_dim: int = 512) -> None:
        super(ToStyleHead, self).__init__()

        self.out_dim = output_dim
        self.convs = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(input_dim, affine=True),
            nn.PReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_dim, output_dim, kernel_size=1)
        )

    def forward(self, x):
        x = self.convs(x)

        return x


class SAN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(SAN, self).__init__()

        self.encoder = ToStyleHead(input_dim=input_dim, output_dim=768)
        self.code_encode_mean = nn.Linear(1920, 1024)
        self.act_code_encode_mean = nn.PReLU()

        self.code_encode_mean_nl = nn.Linear(1024, 512)
        self.act_code_encode_mean_nl = nn.PReLU()

        self.code_encode_mean_nl2 = nn.Linear(512, 2 * output_dim)

        self.fc_mean_enc = nn.Linear(768, 2 * output_dim)
        self.act_fc_mean_enc = nn.PReLU()

        self.fc_mean = nn.Linear(output_dim, output_dim, bias=False)

        self.fc_std_enc = nn.Linear(768, output_dim)
        self.act_fc_std_enc = nn.PReLU()

        self.fc_std = nn.Linear(output_dim, output_dim, bias=False)

        self.fc_mean_nl2 = nn.Linear(output_dim * 4, output_dim)
        self.act_fc_mean_nl2 = nn.PReLU()

        self.fc_std_nl2= nn.Linear(output_dim * 4, output_dim)
        self.act_fc_std_nl2 = nn.PReLU()

    def forward(self, x, code):
        x = self.encoder(x)
        x = torch.flatten(x, 1)

        code_c_mean = self.act_code_encode_mean(self.code_encode_mean(code))
        code_c_mean = self.act_code_encode_mean_nl(self.code_encode_mean_nl(code_c_mean))
        code_c_mean = torch.tanh(self.code_encode_mean_nl2(code_c_mean))

        x_mean = self.act_fc_mean_enc(self.fc_mean_enc(x))
        
        merge = torch.cat([x_mean, code_c_mean], dim=1)

        mean = self.act_fc_mean_nl2(self.fc_mean_nl2(merge))
        std = self.act_fc_std_nl2(self.fc_std_nl2(merge))

        mean = self.fc_mean(mean)
        std = self.fc_std(std)

        return mean, std


class AdaIN_SET(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, content, style_mean, style_std):
        assert style_mean is not None
        assert style_std is not None

        size = content.size()

        content_mean, content_std = calc_mean_std(content)

        style_mean = style_mean.reshape(size[0], content_mean.shape[1], 1, 1)
        style_std = style_std.reshape(size[0], content_mean.shape[1], 1, 1)
        
        normalized_feat = (content - content_mean.expand(size)) / content_std.expand(size)
        sum_mean = style_mean.expand(size)
        sum_std = style_std.expand(size)
        return normalized_feat * sum_std + sum_mean


class ActNorm(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)
            
        return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc
