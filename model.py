import torch
import torch.nn as nn

"""
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride)
Every conv is a same convolution.
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""

config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs) # tại sao hàm conv2d lại cần in_channels, sẽ ra sao nếu lúc gọi ko thêm kernel_size? Sao lúc khởi tạo lại có số channel đầu vào mà không có kernal_size?
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act # có dùng batch normalization và activation function hay không

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1): # chỉ khác cái CNNBlock bình thường ở cái num_repeats
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1),# việc giảm channel rồi tăng lại thì có ích gì?
                ) # Đây là một hàm với đầu ra là các feature vừa đc trích xuất
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x): # residual block là gì, khi nào mới gọi hàm này?
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x) # feature cũ + feature mới// làm như này thì đc j
            else:
                x = layer(x) # feature mới

        return x

class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(
                2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1 # đầu ra của nó các cell, mỗi cell sẽ chứa thông tin của num_classes và bounding box,
                # * 3 bởi vì dành cho 3 anchor
                # [B, 75, 13, 13]
            ),
        )
        self.num_classes = num_classes

    def forward(self, x): # the output of this model in here
        return (
            self.pred(x) # [B, 75, 13, 13]
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3]) # x.shape[0] là batch_size, tức số lượng ảnh đưa vào model một lần #[B, 3, 25, 13, 13]
            # reshape biến mảng 1 chiều [75] thành mảng 2 chiều [3, 25]
            .permute(0, 1, 3, 4, 2) # để sắp xếp lại thứ tự của output # [B, 3, 13, 13, 25]
        )



class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=80):
        super().__init__()
        self.num_classes = num_classes # số class cần dự đoán?
        self.in_channels = in_channels # in channels là chiều sâu đầu vào
        self.layers = self._create_conv_layers() # tạo ra các layer dựa trên kiến trúc YOLOv3

    def forward(self, x):
        outputs = []  # for each scale
        route_connections = [] # to save feature maps for concatenation
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x)) # output lưu dự đoán của mỗi scale (mỗi scale gồm dự đoán của từng cell, mỗi cell gồm dự đoán cho 3 anchor)
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8: # lưu các feature cần concatenate
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs

    def _create_conv_layers(self): # create conv layer from config file
        layers = nn.ModuleList()
        in_channels = self.in_channels # ban đầu số channel đầu vào là số chiều của ảnh

        for module in config: # lấy các số từ config
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels, # cũng là số kernel sử dụng để trích xuất feature map
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels # sau đấy số channel đầu ra (tức số lượng filter) là số channel đầu vào của vòng lặp sau

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats,))

            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes), # thủ tục trước khi đưa ra dự đoán
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2),) # dùng upsample để tăng resolution
                    in_channels = in_channels * 3 # sau khi upsample, số resolution của feature map sẽ bằng so với feature map trước đấy.
                    # muốn biết tại sao số channels *3 thì nhìn vào phần config

        return layers


if __name__ == "__main__":
    num_classes = 20
    IMAGE_SIZE = 416
    model = YOLOv3(num_classes=num_classes)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE)) # 2 là batch size, 3 là số channel của ảnh, IMAGE_SIZE là chiều cao và chiều rộng của ảnh
    out = model(x)
    assert model(x)[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5) # dùng assert để kiểm tra xem output của model có đúng shape hay không, nếu không đúng thì sẽ báo lỗi
    assert model(x)[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    assert model(x)[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
    print("Success!")

"""
- Đầu vào gọi YOLOv3 với tham số là số class
- Hàm YOLOv3 cần 2 tham số là số channel của ảnh(kiểu j cũng là 3) và số class
- Hàm YOLOv3 sẽ tự khởi tạo ra các layer dựa trên kiến trúc của YOLOv3
- Gọi model(x), hàm forward sẽ được gọi với x được truyền vào
- Hàm forward có nhiệm vụ lưu các dự đoán mỗi scale, lưu các feature cần concatenate, và concatenate
"""