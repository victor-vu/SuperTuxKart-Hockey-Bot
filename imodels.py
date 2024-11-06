import torch
import torch.nn.functional as F


def spatial_argmax(logit):
    """
    Compute the soft-argmax of a heatmap
    :param logit: A tensor of size BS x H x W
    :return: A tensor of size BS x 2 the soft-argmax in normalized coordinates (-1 .. 1)
    RETURN now (y,x)
    """
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    return torch.stack(
        (
            (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1),
            (weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
        ),
        1,
    )


def extract_peak(heatmap, max_pool_ks=7, min_score=-5, max_det=100):
    """
    Your code here.
    Extract local maxima (peaks) in a 2d heatmap.
    @heatmap: H x W heatmap containing peaks (similar to your training heatmap)
    @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
    @min_score: Only return peaks greater than min_score
    @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
             heatmap value at the peak. Return no more than max_det peaks per image
    """
    max_cls = F.max_pool2d(heatmap[None, None], kernel_size=max_pool_ks, padding=max_pool_ks // 2, stride=1)[0, 0]
    possible_det = heatmap - (max_cls > heatmap).float() * 1e5
    if max_det > possible_det.numel():
        max_det = possible_det.numel()
    score, loc = torch.topk(possible_det.view(-1), max_det)
    return [(float(s), int(l) % heatmap.size(1), int(l) // heatmap.size(1)) for s, l in zip(score.cpu(), loc.cpu()) if s > min_score]


class norm_layer(torch.nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        self.norm = torch.nn.InstanceNorm2d(self.num_channels)

    def forward(self, x):
        if x.size(2) != 1 and x.size(3) != 1:
            x = self.norm(x)
        return x


def get_norm(norm_name, num_channels):
    if norm_name == "instance":
        return norm_layer(num_channels)
    else:
        return torch.nn.BatchNorm2d(num_channels)


class norm_with_convolution(torch.nn.Module):
    def __init__(self, in_ch, out_ch, norm_type="no_instance", act_name="SiLU", kernel_size=3):
        super().__init__()
        self.norm = get_norm(norm_type, in_ch)
        self.conv = torch.nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.act = torch.nn.SiLU()

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        x = self.act(x)
        return x


class Block(torch.nn.Module):
    def __init__(self, n_input, n_output, kernel_size=3, stride=2, dilation=1):
        super().__init__()
        self.c1 = torch.nn.Conv2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2, stride=stride, dilation=1)
        self.c2 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2)
        self.c3 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2)
        self.b1 = torch.nn.BatchNorm2d(n_output)
        self.b2 = torch.nn.BatchNorm2d(n_output)
        self.b3 = torch.nn.BatchNorm2d(n_output)
        self.skip = torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride)

    def forward(self, x):
        return F.silu(self.b3(self.c3(F.silu(self.b2(self.c2(F.silu(self.b1(self.c1(x)))))))) + self.skip(x))


class UpBlock(torch.nn.Module):
    def __init__(self, n_input, n_output, kernel_size=3, stride=2):
        super().__init__()
        self.c1 = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2, stride=stride, output_padding=1)
        self.conv_with_norm1 = norm_with_convolution(n_output, n_output)
        self.conv_with_norm2 = norm_with_convolution(n_output, n_output)

    def forward(self, x):
        x = F.silu(self.c1(x))
        x = self.conv_with_norm1(x)
        x = self.conv_with_norm2(x)
        return x


class smallBlock(torch.nn.Module):
    def __init__(self, n_input, n_output, kernel_size=3, stride=2, dilation=1):
        super().__init__()
        self.c1 = torch.nn.Conv2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2, stride=stride, dilation=1)
        self.c2 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2)
        self.b1 = torch.nn.BatchNorm2d(n_output)
        self.b2 = torch.nn.BatchNorm2d(n_output)
        self.skip = torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride)

    def forward(self, x):
        return F.silu(self.b2(self.c2(F.silu(self.b1(self.c1(x)))))) + self.skip(x)


class smallUpBlock(torch.nn.Module):
    def __init__(self, n_input, n_output, kernel_size=3, stride=2):
        super().__init__()
        self.c1 = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2, stride=stride, output_padding=1)
        self.conv_with_norm1 = norm_with_convolution(n_output, n_output)

    def forward(self, x):
        x = F.silu(self.c1(x))
        x = self.conv_with_norm1(x)
        return x


class imodel1(torch.nn.Module):
    def __init__(self, layers=[32, 64, 96, 128], n_class=1, kernel_size=3, use_skip=True):
        super().__init__()
        self.input_mean = torch.Tensor([0.4603, 0.5498, 0.6117])
        self.input_std = torch.Tensor([0.2129, 0.1936, 0.1838])

        c = 3
        self.use_skip = use_skip
        self.n_conv = len(layers)
        self.backbone = torch.nn.ModuleDict()  # Dictionary to hold all layers
        skip_layer_size = [3] + layers[:-1]

        # Adding down-sampling layers
        for i, l in enumerate(layers):
            self.backbone[f"conv{i}"] = Block(c, l, kernel_size, 2, dilation=2)
            c = l

        # Adding up-sampling layers
        for i, l in reversed(list(enumerate(layers))):
            self.backbone[f"upconv{i}"] = UpBlock(c, l // 2, kernel_size, 2)
            c = l // 2
            if self.use_skip:
                c += skip_layer_size[i]
        self.silu = torch.nn.SiLU()
        # self.c_b1 = torch.nn.BatchNorm2d(16)
        # self.c_b2 = torch.nn.BatchNorm2d(16)
        # self.c_conv1 = torch.nn.Conv2d(c, 16, kernel_size=3, stride=1, padding=1)
        # self.c_conv2 = torch.nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        # self.c_conv1x1 = torch.nn.Conv2d(16, 1, 1)

        self.c_part = torch.nn.ModuleDict(
            {
                "silu": torch.nn.SiLU(),
                "c_b1": torch.nn.BatchNorm2d(16),
                "c_b2": torch.nn.BatchNorm2d(16),
                "c_conv1": torch.nn.Conv2d(c, 16, kernel_size=3, stride=1, padding=1),
                "c_conv2": torch.nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
                "c_conv1x1": torch.nn.Conv2d(16, 1, 1),
            }
        )

        # self.d_b1 = torch.nn.BatchNorm2d(16)
        # self.d_b2 = torch.nn.BatchNorm2d(16)
        # self.d_b3 = torch.nn.BatchNorm2d(16)
        # self.d_b4 = torch.nn.BatchNorm2d(16)

        # self.d_conv1 = torch.nn.Conv2d(c, 16, kernel_size=3, stride=2, padding=1, dilation=2)
        # self.d_conv2 = torch.nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        # self.d_conv3 = torch.nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        # self.d_conv4 = torch.nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)

        # self.d_f1 = torch.nn.Linear(16 * 8 * 11, 512)
        # self.d_f2 = torch.nn.Linear(512, 256)
        # self.d_f3 = torch.nn.Linear(256, 1)

        self.d_part = torch.nn.ModuleDict(
            {
                "d_b1": torch.nn.BatchNorm2d(16),
                "d_b2": torch.nn.BatchNorm2d(16),
                "d_b3": torch.nn.BatchNorm2d(16),
                "d_b4": torch.nn.BatchNorm2d(16),
                "d_conv1": torch.nn.Conv2d(c, 16, kernel_size=3, stride=2, padding=1, dilation=2),
                "d_conv2": torch.nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
                "d_conv3": torch.nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
                "d_conv4": torch.nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
                "d_f1": torch.nn.Linear(16 * 9 * 12, 416),  # 16 * 8 * 11
                "d_f2": torch.nn.Linear(416, 256),
                "d_f3": torch.nn.Linear(256, 1),
                "silu": torch.nn.SiLU(),
                "d_dropout1": torch.nn.Dropout(p=0.2),
                "d_dropout2": torch.nn.Dropout(p=0.2),
            }
        )
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.f_f4 = torch.nn.Linear(128 + 19 + 1728, 416)
        self.f_f5 = torch.nn.Linear(416, 256)
        self.f_f6 = torch.nn.Linear(256, 1)
        self.f_dropout45 = torch.nn.Dropout(p=0.25)
        self.f_dropout56 = torch.nn.Dropout(p=0.25)

    def predict(self, x):
        z = (x - self.input_mean[None, :, None, None].to(x.device)) / self.input_std[None, :, None, None].to(x.device)

        # Forward pass through the down-sampling layers
        # skip_connections = []
        # for i in range(self.n_conv):
        #    if self.use_skip:
        #        skip_connections.append(z)
        #    z = self.backbone[f"conv{i}"](z)
        # z_middle = z
        ## Forward pass through the up-sampling layers
        # for i in reversed(range(self.n_conv)):
        #    z = self.backbone[f"upconv{i}"](z)
        #    z = z[:, :, : skip_connections[i].size(2), : skip_connections[i].size(3)]  # Fix padding
        #    if self.use_skip:
        #        z = torch.cat([z, skip_connections[i]], dim=1)
        skip_connections = []

        # First loop
        if self.use_skip:
            skip_connections.append(z)
        z = self.backbone["conv0"](z)

        if self.use_skip:
            skip_connections.append(z)
        z = self.backbone["conv1"](z)

        if self.use_skip:
            skip_connections.append(z)
        z = self.backbone["conv2"](z)
        if self.use_skip:
            skip_connections.append(z)
        z = self.backbone["conv3"](z)

        z_middle = z

        # Second loop
        z = self.backbone["upconv3"](z)
        z = z[:, :, : skip_connections[2].size(2), : skip_connections[2].size(3)]  # Fix padding
        if self.use_skip:
            z = torch.cat([z, skip_connections[3]], dim=1)
        z = self.backbone["upconv2"](z)
        z = z[:, :, : skip_connections[2].size(2), : skip_connections[2].size(3)]  # Fix padding
        if self.use_skip:
            z = torch.cat([z, skip_connections[2]], dim=1)

        z = self.backbone["upconv1"](z)
        z = z[:, :, : skip_connections[1].size(2), : skip_connections[1].size(3)]  # Fix padding
        if self.use_skip:
            z = torch.cat([z, skip_connections[1]], dim=1)

        z = self.backbone["upconv0"](z)
        z = z[:, :, : skip_connections[0].size(2), : skip_connections[0].size(3)]  # Fix padding
        if self.use_skip:
            z = torch.cat([z, skip_connections[0]], dim=1)

        # print(z.shape)

        c = z
        c = self.c_part["c_conv1"](z)
        c = self.c_part["c_b1"](c)
        c = self.silu(c)
        c = self.c_part["c_conv2"](c)
        c = self.c_part["c_b2"](c)
        c = self.silu(c)
        c = self.c_part["c_conv1x1"](c)
        # for layer in ["c_conv1", "c_b1", "silu", "c_conv2", "c_b2", "silu", "c_conv1x1"]:
        #    c = self.c_part[layer](c)

        # c = self.c_conv1(z)
        # c = self.c_b1(c)
        # c = self.silu(c)
        # c = self.c_conv2(c)
        # c = self.c_b2(c)
        # c = self.silu(c)
        # c = self.c_conv1x1(c)

        # d = self.d_conv1(z)
        # d = self.d_b1(d)
        # d = self.silu(d)
        # d = self.d_conv2(d)
        # d = self.d_b2(d)
        # d = self.silu(d)
        # d = self.d_conv3(d)
        # d = self.d_b3(d)
        # d = self.silu(d)
        # d = self.d_conv4(d)
        # d = self.d_b4(d)
        # d = self.silu(d)
        # d = d.view(-1, 16 * 8 * 11)
        # d = self.silu(self.d_f1(d))
        # d = self.silu(self.d_f2(d))
        # d = self.d_f3(d)

        d = z
        # for layer in ["d_conv1", "d_b1", "silu", "d_conv2", "d_b2", "silu", "d_conv3", "d_b3", "silu", "d_conv4", "d_b4", "silu"]:
        #    d = self.d_part[layer](d)
        d = self.d_part["d_conv1"](z)
        d = self.d_part["d_b1"](d)
        d = self.silu(d)
        d = self.d_part["d_conv2"](d)
        d = self.d_part["d_b2"](d)
        d = self.silu(d)
        d = self.d_part["d_conv3"](d)
        d = self.d_part["d_b3"](d)
        d = self.silu(d)
        d = self.d_part["d_conv4"](d)
        d = self.d_part["d_b4"](d)
        d = self.silu(d)
        # print("d227",d.shape)

        d2 = d.view(-1, 16 * 9 * 12)
        # print(d2.shape)
        d = self.d_part["d_dropout1"](d2)
        d = self.silu(self.d_part["d_f1"](d))
        d = self.d_part["d_dropout2"](d)
        d = self.silu(self.d_part["d_f2"](d))
        d = self.d_part["d_f3"](d)

        f1 = self.pool(z_middle)
        f2 = self.pool(z)
        f3 = d2  # self.pool(d2)
        # print(f1.shape, f2.shape, f3.shape)
        f = torch.cat([f1.view(f1.size(0), -1), f2.view(f2.size(0), -1), f3.view(f3.size(0), -1)], dim=1)
        # print(f.shape, f1.shape, f2.shape)
        f = self.silu(self.f_f4(f))
        f = self.f_dropout45(f)
        # print(f.shape)
        f = self.silu(self.f_f5(f))
        f = self.f_dropout56(f)
        f = self.f_f6(f)

        return c, d, f

    def forward(self, img):
        """
        Your code here
        Predict the aim point in image coordinate, given the supertuxkart image
        @img: (B,3,96,128)
        return (B,2)
        """

        det, dep, flag = self.predict(img)
        loc = spatial_argmax(det[:, 0, :, :])

        return loc, dep, flag


class imodel2(torch.nn.Module):
    def __init__(self, layers=[32, 64, 96, 128], n_class=1, kernel_size=3, use_skip=True):
        super().__init__()
        self.input_mean = torch.Tensor([0.4603, 0.5498, 0.6117])
        self.input_std = torch.Tensor([0.2129, 0.1936, 0.1838])

        c = 3
        self.use_skip = use_skip
        self.n_conv = len(layers)
        self.backbone = torch.nn.ModuleDict()  # Dictionary to hold all layers
        skip_layer_size = [3] + layers[:-1]

        # Adding down-sampling layers
        for i, l in enumerate(layers):
            self.backbone[f"conv{i}"] = Block(c, l, kernel_size, 2, dilation=2)
            c = l

        # Adding up-sampling layers
        for i, l in reversed(list(enumerate(layers))):
            self.backbone[f"upconv{i}"] = UpBlock(c, l // 2, kernel_size, 2)
            c = l // 2
            if self.use_skip:
                c += skip_layer_size[i]
        self.silu = torch.nn.SiLU()

        self.c_part = torch.nn.ModuleDict(
            {
                "silu": torch.nn.SiLU(),
                "c_b1": torch.nn.BatchNorm2d(16),
                "c_b2": torch.nn.BatchNorm2d(16),
                "c_conv1": torch.nn.Conv2d(c, 16, kernel_size=3, stride=1, padding=1),
                "c_conv2": torch.nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
                "c_conv1x1": torch.nn.Conv2d(16, 1, 1),
            }
        )

        self.d_part = torch.nn.ModuleDict(
            {
                "d_b1": torch.nn.BatchNorm2d(16),
                "d_b2": torch.nn.BatchNorm2d(16),
                "d_b3": torch.nn.BatchNorm2d(16),
                "d_b4": torch.nn.BatchNorm2d(16),
                "d_conv1": torch.nn.Conv2d(c, 16, kernel_size=3, stride=2, padding=1, dilation=2),
                "d_conv2": torch.nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
                "d_conv3": torch.nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
                "d_conv4": torch.nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
                "d_f1": torch.nn.Linear(16 * 8 * 18, 400),  # 16 * 8 * 11
                "d_f2": torch.nn.Linear(400, 256),
                "d_f3": torch.nn.Linear(256, 1),
                "silu": torch.nn.SiLU(),
                "d_dropout1": torch.nn.Dropout(p=0.2),
                "d_dropout2": torch.nn.Dropout(p=0.2),
            }
        )
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.f_f4 = torch.nn.Linear(128 + 19 + 2304, 400)
        self.f_f5 = torch.nn.Linear(400, 256)
        self.f_f6 = torch.nn.Linear(256, 1)
        self.f_dropout45 = torch.nn.Dropout(p=0.25)
        self.f_dropout56 = torch.nn.Dropout(p=0.25)

    def predict(self, x):
        z = (x - self.input_mean[None, :, None, None].to(x.device)) / self.input_std[None, :, None, None].to(x.device)

        skip_connections = []

        # First loop
        if self.use_skip:
            skip_connections.append(z)
        z = self.backbone["conv0"](z)

        if self.use_skip:
            skip_connections.append(z)
        z = self.backbone["conv1"](z)

        if self.use_skip:
            skip_connections.append(z)
        z = self.backbone["conv2"](z)
        if self.use_skip:
            skip_connections.append(z)
        z = self.backbone["conv3"](z)

        z_middle = z

        # Second loop
        z = self.backbone["upconv3"](z)
        z = z[:, :, : skip_connections[2].size(2), : skip_connections[2].size(3)]  # Fix padding
        if self.use_skip:
            z = torch.cat([z, skip_connections[3]], dim=1)
        z = self.backbone["upconv2"](z)
        z = z[:, :, : skip_connections[2].size(2), : skip_connections[2].size(3)]  # Fix padding
        if self.use_skip:
            z = torch.cat([z, skip_connections[2]], dim=1)

        z = self.backbone["upconv1"](z)
        z = z[:, :, : skip_connections[1].size(2), : skip_connections[1].size(3)]  # Fix padding
        if self.use_skip:
            z = torch.cat([z, skip_connections[1]], dim=1)

        z = self.backbone["upconv0"](z)
        z = z[:, :, : skip_connections[0].size(2), : skip_connections[0].size(3)]  # Fix padding
        if self.use_skip:
            z = torch.cat([z, skip_connections[0]], dim=1)

        # print(z.shape)

        c = z
        c = self.c_part["c_conv1"](z)
        c = self.c_part["c_b1"](c)
        c = self.silu(c)
        c = self.c_part["c_conv2"](c)
        c = self.c_part["c_b2"](c)
        c = self.silu(c)
        c = self.c_part["c_conv1x1"](c)

        d = z
        d = self.d_part["d_conv1"](z)
        d = self.d_part["d_b1"](d)
        d = self.silu(d)
        d = self.d_part["d_conv2"](d)
        d = self.d_part["d_b2"](d)
        d = self.silu(d)
        d = self.d_part["d_conv3"](d)
        d = self.d_part["d_b3"](d)
        d = self.silu(d)
        d = self.d_part["d_conv4"](d)
        d = self.d_part["d_b4"](d)
        d = self.silu(d)
        # print("460 ", d.shape)

        d2 = d.view(-1, 16 * 8 * 18)
        # print("463 ", d2.shape)
        d = self.d_part["d_dropout1"](d2)
        d = self.silu(self.d_part["d_f1"](d))
        d = self.d_part["d_dropout2"](d)
        d = self.silu(self.d_part["d_f2"](d))
        d = self.d_part["d_f3"](d)

        f1 = self.pool(z_middle)
        f2 = self.pool(z)
        f3 = d2  # self.pool(d2)
        # print("473 ", f1.shape, f2.shape, f3.shape)
        f = torch.cat([f1.view(f1.size(0), -1), f2.view(f2.size(0), -1), f3.view(f3.size(0), -1)], dim=1)
        f = self.silu(self.f_f4(f))
        f = self.f_dropout45(f)
        # print("477 ", f.shape)
        f = self.silu(self.f_f5(f))
        f = self.f_dropout56(f)
        f = self.f_f6(f)

        return c, d, f

    def forward(self, img):
        """
        Your code here
        Predict the aim point in image coordinate, given the supertuxkart image
        @img: (B,3,96,128)
        return (B,2)
        """

        det, dep, flag = self.predict(img)
        loc = spatial_argmax(det[:, 0, :, :])

        return loc, dep, flag


class imodel3(torch.nn.Module):
    def __init__(self, layers=[32, 48, 64, 92], n_class=1, kernel_size=3, use_skip=True):
        super().__init__()
        self.input_mean = torch.Tensor([0.4603, 0.5498, 0.6117])
        self.input_std = torch.Tensor([0.2129, 0.1936, 0.1838])

        c = 3
        self.use_skip = use_skip
        self.n_conv = len(layers)
        self.backbone = torch.nn.ModuleDict()  # Dictionary to hold all layers
        skip_layer_size = [3] + layers[:-1]

        # Adding down-sampling layers
        for i, l in enumerate(layers):
            self.backbone[f"conv{i}"] = smallBlock(c, l, kernel_size, 2, dilation=2)
            c = l

        # Adding up-sampling layers
        for i, l in reversed(list(enumerate(layers))):
            self.backbone[f"upconv{i}"] = UpBlock(c, l // 2, kernel_size, 2)
            c = l // 2
            if self.use_skip:
                c += skip_layer_size[i]
        self.silu = torch.nn.SiLU()

        self.c_part = torch.nn.ModuleDict(
            {
                "silu": torch.nn.SiLU(),
                "c_b1": torch.nn.BatchNorm2d(32),
                "c_b2": torch.nn.BatchNorm2d(24),
                "c_conv1": torch.nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),
                "c_conv2": torch.nn.Conv2d(32, 24, kernel_size=3, stride=1, padding=1),
                "c_conv1x1": torch.nn.Conv2d(24, 1, 1),
                # "c2_dropout1": torch.nn.Dropout2d(6 / 24),
                "c2_dropout2": torch.nn.Dropout2d(8 / 32),
                "c2_dropout3": torch.nn.Dropout2d(6 / 24),
            }
        )

        self.d_part = torch.nn.ModuleDict(
            {
                "d_b1": torch.nn.BatchNorm2d(24),
                "d_b2": torch.nn.BatchNorm2d(24),
                "d_b3": torch.nn.BatchNorm2d(24),
                "d_b4": torch.nn.BatchNorm2d(24),
                "d_conv1": torch.nn.Conv2d(c, 24, kernel_size=3, stride=2, padding=1, dilation=2),
                "d_conv2": torch.nn.Conv2d(24, 24, kernel_size=3, stride=2, padding=1),
                "d_conv3": torch.nn.Conv2d(24, 24, kernel_size=3, stride=2, padding=1),
                "d_conv4": torch.nn.Conv2d(24, 24, kernel_size=3, stride=2, padding=1),
                "d2_dropout1": torch.nn.Dropout2d(3 / 24),
                "d2_dropout2": torch.nn.Dropout2d(3 / 24),
                "d2_dropout3": torch.nn.Dropout2d(3 / 24),
                "d_f1": torch.nn.Linear(24 * 9 * 12, 1),  # 16 * 8 * 11,
                "d_dropout1": torch.nn.Dropout(p=0.2),
                "silu": torch.nn.SiLU(),
            }
        )
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.f_f4 = torch.nn.Linear(92 + 19 + 2592, 416)
        self.f_f5 = torch.nn.Linear(416, 256)
        self.f_f6 = torch.nn.Linear(256, 1)
        self.f_dropout45 = torch.nn.Dropout(p=0.3)
        self.f_dropout56 = torch.nn.Dropout(p=0.3)

    def predict(self, x):
        z = (x - self.input_mean[None, :, None, None].to(x.device)) / self.input_std[None, :, None, None].to(x.device)

        skip_connections = []

        # First loop
        if self.use_skip:
            skip_connections.append(z)
        z = self.backbone["conv0"](z)

        if self.use_skip:
            skip_connections.append(z)
        z = self.backbone["conv1"](z)

        if self.use_skip:
            skip_connections.append(z)
        z = self.backbone["conv2"](z)
        if self.use_skip:
            skip_connections.append(z)
        z = self.backbone["conv3"](z)

        z_middle = z

        # Second loop
        z = self.backbone["upconv3"](z)
        z = z[:, :, : skip_connections[2].size(2), : skip_connections[2].size(3)]  # Fix padding
        if self.use_skip:
            z = torch.cat([z, skip_connections[3]], dim=1)
        z = self.backbone["upconv2"](z)
        z = z[:, :, : skip_connections[2].size(2), : skip_connections[2].size(3)]  # Fix padding
        if self.use_skip:
            z = torch.cat([z, skip_connections[2]], dim=1)

        z = self.backbone["upconv1"](z)
        z = z[:, :, : skip_connections[1].size(2), : skip_connections[1].size(3)]  # Fix padding
        if self.use_skip:
            z = torch.cat([z, skip_connections[1]], dim=1)

        z = self.backbone["upconv0"](z)
        z = z[:, :, : skip_connections[0].size(2), : skip_connections[0].size(3)]  # Fix padding
        if self.use_skip:
            z = torch.cat([z, skip_connections[0]], dim=1)

        # print(z.shape)

        c = z
        c = self.c_part["c_conv1"](z)
        c = self.c_part["c_b1"](c)
        c = self.c_part["c2_dropout2"](c)
        c = self.silu(c)
        c = self.c_part["c_conv2"](c)
        c = self.c_part["c_b2"](c)
        c = self.c_part["c2_dropout3"](c)
        c = self.silu(c)
        c = self.c_part["c_conv1x1"](c)

        d = z
        d = self.d_part["d_conv1"](z)
        d = self.d_part["d_b1"](d)
        d = self.d_part["d2_dropout1"](d)
        d = self.silu(d)
        d = self.d_part["d_conv2"](d)
        d = self.d_part["d_b2"](d)
        d = self.d_part["d2_dropout2"](d)
        d = self.silu(d)
        d = self.d_part["d_conv3"](d)
        d = self.d_part["d_b3"](d)
        d = self.d_part["d2_dropout3"](d)
        d = self.silu(d)
        d = self.d_part["d_conv4"](d)
        d = self.d_part["d_b4"](d)
        d = self.silu(d)
        # print("d227", d.shape)

        d2 = d.view(-1, 24 * 9 * 12)
        # print(d2.shape)
        d = self.d_part["d_dropout1"](d2)
        d = self.silu(self.d_part["d_f1"](d))

        f1 = self.pool(z_middle)
        f2 = self.pool(z)
        f3 = d2  # self.pool(d2)
        # print(f1.shape, f2.shape, f3.shape)
        f = torch.cat([f1.view(f1.size(0), -1), f2.view(f2.size(0), -1), f3.view(f3.size(0), -1)], dim=1)
        # print(f.shape, f1.shape, f2.shape)
        f = self.silu(self.f_f4(f))
        f = self.f_dropout45(f)
        # print(f.shape)
        f = self.silu(self.f_f5(f))
        f = self.f_dropout56(f)
        f = self.f_f6(f)

        return c, d, f

    def forward(self, img):
        """
        Your code here
        Predict the aim point in image coordinate, given the supertuxkart image
        @img: (B,3,96,128)
        return (B,2)
        """

        det, dep, flag = self.predict(img)
        loc = spatial_argmax(det[:, 0, :, :])

        return loc, dep, flag


def save_model(model, name="imodel.th"):
    from torch import save
    from os import path

    if isinstance(model, imodel1) or isinstance(model, imodel2):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), name))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model():
    from torch import load
    from os import path

    r = imodel()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), "planner.th"), map_location="cpu"))
    return r
