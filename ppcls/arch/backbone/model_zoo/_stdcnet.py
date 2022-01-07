# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F

import math
import sys


from ppcls.utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url

__all__ = ["STDCNet_1", "STDCNet_2", "STDCNet_3"]

class param_init():
    @staticmethod
    def constant_init(param, **kwargs):
        initializer = nn.initializer.Constant(**kwargs)
        initializer(param, param.block)

    @staticmethod
    def normal_init(param, **kwargs):
        initializer = nn.initializer.Normal(**kwargs)
        initializer(param, param.block)

"""
class STDCNet(nn.Layer):
    def __init__(self,
                 base=64,
                 layers=[4, 5, 3],
                 block_num=4,
                 block_type="cat",
                 num_classes=1000,
                 dropout=0.20,
                 use_conv_last=False,
                 pretrained=None):
        super().__init__()

        block_dict = {"cat": CatBottleneck,
                      "add": AddBottleneck,
                      "dilation_2": CatBottleneck_dilation_2,
                      "gap_no_conv": CatBottleneck_gap_no_conv,
                      "shuffle": CatBottleneck_shuffle,
                      "split": CatBottleneck_split,
                      "repvgg": None,
                      "last_se": CatBottleneck_last_se,
                      "dilation_pool": CatBottleneck_dilation_pool,
                      "dw": CatBottleneck_dw,
                      "last_pool": CatBottleneck_last_pool,
                      "dilation_1": CatBottleneck_dilation_1,
                      "dw_pool": CatBottleneck_dw_pool,
                      "dw_dilation_pool": CatBottleneck_dw_dilation_pool}

        print(block_type)
        assert block_type in block_dict
        block = block_dict[block_type]
        self.feat_channels = [1024]

        self.use_conv_last = use_conv_last
        self.features = self._make_layers(base, layers, block_num, block)
        self.conv_last = ConvBNRelu(base * 16, max(1024, base * 16), 1, 1)

        self.fc = nn.Linear(max(1024, base*16), max(1024, base*16), bias_attr=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(max(1024, base*16), num_classes, bias_attr=False)

        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        '''
        forward function for feature extract.
        '''
        out = self.features(x)
        out = self.conv_last(out).pow(2)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.flatten(1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear(out)
        return out

    def _make_layers(self, base, layers, block_num, block):
        features = []
        features += [ConvBNRelu(3, base // 2, 3, 2)]
        features += [ConvBNRelu(base // 2, base, 3, 2)]

        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base * 4, block_num, 2))
                elif j == 0:
                    features.append(
                        block(base * int(math.pow(2, i + 1)),
                              base * int(math.pow(2, i + 2)), block_num, 2))
                else:
                    features.append(
                        block(base * int(math.pow(2, i + 2)),
                              base * int(math.pow(2, i + 2)), block_num, 1))

        return nn.Sequential(*features)

    def init_weight(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                param_init.normal_init(layer.weight, std=0.001)
            elif isinstance(layer, (nn.BatchNorm, nn.SyncBatchNorm)):
                param_init.constant_init(layer.weight, value=1.0)
                param_init.constant_init(layer.bias, value=0.0)
        if self.pretrained is not None:
            utils.load_pretrained_model(self, self.pretrained)


class STDCNet_slim(nn.Layer):
    def __init__(self,
                 base=64,
                 layers=[4, 5, 3],
                 block_num=4,
                 block_type="shuffle_simple",
                 num_classes=1000,
                 dropout=0.20,
                 use_conv_last=False,
                 pretrained=None):
        super().__init__()
        block_dict = {"cat": CatBottleneck,
                      "add": AddBottleneck,
                      "dilation_1": CatBottleneck_dilation_1,
                      "dilation_2": CatBottleneck_dilation_2,
                      "gap_no_conv": CatBottleneck_gap_no_conv,
                      "shuffle": CatBottleneck_shuffle,
                      "split": CatBottleneck_split,
                      "repvgg": None,
                      "last_pool": CatBottleneck_last_pool,
                      "last_se": CatBottleneck_last_se,
                      "dilation_pool": CatBottleneck_dilation_pool,
                      "dw": CatBottleneck_dw, 
                      "shuffle_simple": CatBottleneck_shuffle_simple}
        assert block_type in block_dict
        block = block_dict[block_type]
        self.feat_channels = [512]

        self.features = self._make_layers(base, layers, block_num, block)
        
        self.x2 = nn.Sequential(self.features[:1])
        self.x4 = nn.Sequential(self.features[1:2])
        idx = (2 + layers[0], 2 + layers[0] + layers[1])
        self.x8 = nn.Sequential(self.features[2:idx[0]])
        self.x16 = nn.Sequential(self.features[idx[0]:idx[1]])
        self.x32 = nn.Sequential(self.features[idx[1]:])

        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        feat2 = self.x2(x)
        feat4 = self.x4(feat2)
        feat8 = self.x8(feat4)
        feat16 = self.x16(feat8)
        feat32 = self.x32(feat16)
        return feat2, feat4, feat8, feat16, feat32

    def _make_layers(self, base, layers, block_num, block):
        features = []
        features += [ConvBNRelu(3, base // 2, 3, 2)]
        features += [ConvBNRelu(base // 2, base, 3, 2)]

        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base * 2, block_num, 2))
                elif j == 0:
                    features.append(
                        block(base * int(math.pow(2, i)),
                              base * int(math.pow(2, i + 1)), block_num, 2))
                else:
                    features.append(
                        block(base * int(math.pow(2, i + 1)),
                              base * int(math.pow(2, i + 1)), block_num, 1))

        return nn.Sequential(*features)

    def init_weight(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                param_init.normal_init(layer.weight, std=0.001)
            elif isinstance(layer, (nn.BatchNorm, nn.SyncBatchNorm)):
                param_init.constant_init(layer.weight, value=1.0)
                param_init.constant_init(layer.bias, value=0.0)
        if self.pretrained is not None:
            utils.load_pretrained_model(self, self.pretrained)

class ConvBNRelu(nn.Layer):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1, dilation=1, act=nn.ReLU, groups=1):
        super(ConvBNRelu, self).__init__()
        self.conv = nn.Conv2D(
            in_planes,
            out_planes,
            kernel_size=kernel,
            stride=stride,
            padding=kernel // 2 if dilation == 1 else dilation,
            dilation=dilation,
            bias_attr=False,
            groups=groups)
        self.bn = nn.SyncBatchNorm(out_planes, data_format='NCHW')
        self.act = act()

    def forward(self, x):
        out = self.act(self.bn(self.conv(x)))
        return out


class AddBottleneck(nn.Layer):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(AddBottleneck, self).__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.LayerList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias_attr=False),
                nn.BatchNorm2D(out_planes // 2),
            )
            self.skip = nn.Sequential(
                nn.Conv2D(
                    in_planes,
                    in_planes,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=in_planes,
                    bias_attr=False),
                nn.BatchNorm2D(in_planes),
                nn.Conv2D(
                    in_planes, out_planes, kernel_size=1, bias_attr=False),
                nn.BatchNorm2D(out_planes),
            )
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx))))

    def forward(self, x):
        out_list = []
        out = x
        for idx, conv in enumerate(self.conv_list):
            if idx == 0 and self.stride == 2:
                out = self.avd_layer(conv(out))
            else:
                out = conv(out)
            out_list.append(out)
        if self.stride == 2:
            x = self.skip(x)
        return paddle.concat(out_list, axis=1) + x


class CatBottleneck(nn.Layer):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(CatBottleneck, self).__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.LayerList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias_attr=False),
                nn.BatchNorm2D(out_planes // 2),
            )
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx))))

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        out = paddle.concat(out_list, axis=1)
        return out


class CatBottleneck_avg_4_block(nn.Layer):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super().__init__()
        assert block_num in (2, 4)
        assert stride in (1, 2)

        self.conv_list = nn.LayerList()
        self.stride = stride
        if stride == 2:
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)

        mid_chs = out_planes // block_num
        if block_num == 2:
            self.conv_list.append(
                    ConvBNRelu(in_planes, mid_chs, kernel=2, stride=1))
            self.conv_list.append(
                    ConvBNRelu(mid_chs, mid_chs, kernel=3, stride=stride))
        elif block_num == 4:
            self.conv_list.append(
                    ConvBNRelu(in_planes, mid_chs, kernel=3, stride=1))
            self.conv_list.append(
                    ConvBNRelu(mid_chs, mid_chs, kernel=3, stride=stride))
            self.conv_list.append(
                    ConvBNRelu(mid_chs, mid_chs, kernel=3, stride=1))
            self.conv_list.append(
                    ConvBNRelu(mid_chs, mid_chs, kernel=3, stride=1))

    def forward(self, x):
        out_list = []

        out = self.conv_list[0](x)
        if self.stride == 1:
            out_list.insert(0, out)
        else:
            out1 = self.skip(out)
            out_list.insert(0, out1)

        for _, conv in enumerate(self.conv_list[1:]):
            out = conv(out)
            out_list.append(out)

        out = paddle.concat(out_list, axis=1)
        return out


class CatBottleneck_add_short_cut(nn.Layer):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super().__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.LayerList()
        self.stride = stride

        self.short_cut = False
        if in_planes == out_planes and stride == 1:
            self.short_cut = True

        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias_attr=False),
                nn.BatchNorm2D(out_planes // 2),
            )
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx))))

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        out = paddle.concat(out_list, axis=1)
        if self.short_cut:
            out = out + x
        return out
    

class CatBottleneck_remove_dwconv(nn.Layer):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super().__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.LayerList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias_attr=False),
                nn.BatchNorm2D(out_planes // 2),
            )
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx))))

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        out = paddle.concat(out_list, axis=1)
        return out


class CatBottleneck_new_act(nn.Layer):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super().__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.LayerList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias_attr=False),
                nn.BatchNorm2D(out_planes // 2),
            )
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)
            stride = 1

        act = nn.PReLU
        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(in_planes, out_planes // 2, kernel=1, act=act))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 2, stride=stride, act=act))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 4, stride=stride, act=act))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx + 1)), act=act))
            else:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx)), act=act))

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        out = paddle.concat(out_list, axis=1)
        return out

class CatBottleneck_dilation_1(nn.Layer):
    '''the last one is dilation conv'''
    def __init__(self, in_planes, out_planes, block_num, stride=1):
        super().__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.LayerList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias_attr=False),
                nn.BatchNorm2D(out_planes // 2),
            )
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx)), dilation=2))

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        out = paddle.concat(out_list, axis=1)
        return out

class CatBottleneck_dilation_2(nn.Layer):
    '''the last two is dilation conv'''
    def __init__(self, in_planes, out_planes, block_num, stride=1):
        super().__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.LayerList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias_attr=False),
                nn.BatchNorm2D(out_planes // 2),
            )
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 4, stride=stride))
            elif idx == block_num - 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx + 1)), dilation=2))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx)), dilation=2))

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        out = paddle.concat(out_list, axis=1)
        return out

class CatBottleneck_gap_no_conv(nn.Layer):
    '''add global avg pooling in the last'''
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super().__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.LayerList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias_attr=False),
                nn.BatchNorm2D(out_planes // 2),
            )
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx))))

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)

            if idx == len(self.conv_list[1:]) - 1:
                avg = F.adaptive_avg_pool2d(out, 1)
                out = out + avg
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        out = paddle.concat(out_list, axis=1)
        return out

class CatBottleneck_gap_conv(nn.Layer):
    '''add global avg pooling in the last'''
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super().__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.LayerList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias_attr=False),
                nn.BatchNorm2D(out_planes // 2),
            )
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx))))
        ch = out_planes // int(math.pow(2, idx))
        self.conv = ConvBNRelu(ch, ch, kernel=1)

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)

            if idx == len(self.conv_list[1:]) - 1:
                avg = F.adaptive_avg_pool2d(out, 1)
                avg = self.conv(avg)
                out = out + avg
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        out = paddle.concat(out_list, axis=1)
        return out

class CatBottleneck_shuffle(nn.Layer):
    def __init__(self, in_planes, out_planes, block_num=4, stride=1):
        super().__init__()
        assert block_num == 4, "block number should be larger than 1."
        assert stride in (1, 2)

        self.out_planes = out_planes
        self.stride = stride
        self.conv_list = nn.LayerList()

        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias_attr=False),
                nn.BatchNorm2D(out_planes // 2),
            )
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)

            for idx in range(block_num):
                if idx == 0:
                    self.conv_list.append(
                        ConvBNRelu(in_planes, out_planes // 2, kernel=1))
                elif idx < block_num - 1:
                    self.conv_list.append(
                        ConvBNRelu(out_planes // int(math.pow(2, idx)),
                                out_planes // int(math.pow(2, idx + 1))))
                else:
                    self.conv_list.append(
                        ConvBNRelu(out_planes // int(math.pow(2, idx)),
                                out_planes // int(math.pow(2, idx))))
        else:
            assert in_planes == out_planes
            self.conv_list.append(
                ConvBNRelu(in_planes // 2, in_planes // 4))
            self.conv_list.append(
                ConvBNRelu(in_planes // 4, in_planes // 8))
            self.conv_list.append(
                ConvBNRelu(in_planes // 8, in_planes // 8))
    
    @staticmethod
    def channel_shuffle(x, groups, out_channels=None):
        x_shape = paddle.shape(x)
        if out_channels is None:
            out_channels = x_shape[1]
        channels_per_group = x_shape[1] // groups
        x = paddle.reshape(
            x=x, shape=[x_shape[0], groups, channels_per_group, x_shape[2], x_shape[3]])
        x = paddle.transpose(x=x, perm=[0, 2, 1, 3, 4])
        x = paddle.flatten(x, start_axis=1, stop_axis=2)
        x = paddle.reshape(x=x, shape=[0, out_channels, 0, 0])
        return x

    def forward(self, x):
        if self.stride == 2:
            out_list = []
            out1 = self.conv_list[0](x)
            for idx, conv in enumerate(self.conv_list[1:]):
                if idx == 0:
                    if self.stride == 2:
                        out = conv(self.avd_layer(out1))
                    else:
                        out = conv(out1)
                else:
                    out = conv(out)
                out_list.append(out)

            if self.stride == 2:
                out1 = self.skip(out1)
            out_list.insert(0, out1)
            out = paddle.concat(out_list, axis=1)
            return out
        else:
            x1, x2 = paddle.split(x, num_or_sections=2, axis=1)
            out_list = []
            for conv in self.conv_list:
                x1 = conv(x1)
                out_list.append(x1)
            out_list.append(x2)
            out = paddle.concat(out_list, axis=1)
            out = self.channel_shuffle(out, 2, self.out_planes)
            return out


class CatBottleneck_shuffle_simple(nn.Layer):
    def __init__(self, in_planes, out_planes, block_num=4, stride=1):
        super().__init__()
        assert block_num == 4, "block number should be larger than 1."
        assert (stride == 1 and in_planes == out_planes) \
            or (stride == 2 and 2 * in_planes == out_planes)

        self.out_planes = out_planes
        self.stride = stride
        self.conv_list = nn.LayerList()

        if stride == 1:
            self.conv_list.append(
                ConvBNRelu(in_planes // 2, in_planes // 4))
            self.conv_list.append(
                ConvBNRelu(in_planes // 4, in_planes // 8))
            self.conv_list.append(
                ConvBNRelu(in_planes // 8, in_planes // 8))
        else:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    in_planes,
                    in_planes,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=in_planes,
                    bias_attr=False),
                nn.BatchNorm2D(in_planes),
            )
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)

            self.conv_list.append(
                ConvBNRelu(in_planes, in_planes // 2))
            self.conv_list.append(
                ConvBNRelu(in_planes // 2, in_planes // 4))
            self.conv_list.append(
                ConvBNRelu(in_planes // 4, in_planes // 4))
            
    @staticmethod
    def channel_shuffle(x, groups, out_channels=None):
        x_shape = paddle.shape(x)
        if out_channels is None:
            out_channels = x_shape[1]
        channels_per_group = x_shape[1] // groups
        x = paddle.reshape(
            x=x, shape=[x_shape[0], groups, channels_per_group, x_shape[2], x_shape[3]])
        x = paddle.transpose(x=x, perm=[0, 2, 1, 3, 4])
        x = paddle.flatten(x, start_axis=1, stop_axis=2)
        x = paddle.reshape(x=x, shape=[0, out_channels, 0, 0])
        return x

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = paddle.split(x, num_or_sections=2, axis=1)
            out_list = []
            for conv in self.conv_list:
                x1 = conv(x1)
                out_list.append(x1)
            out_list.append(x2)
            out = paddle.concat(out_list, axis=1)
            out = self.channel_shuffle(out, 2, self.out_planes)
            return out
        else:
            out_list = []
            out1 = self.conv_list[0](self.avd_layer(x))
            out_list.append(out1)
            for idx, conv in enumerate(self.conv_list[1:]):
                out1 = conv(out1)
                out_list.append(out1)

            out2 = self.skip(x)
            out_list.append(out2)
            out = paddle.concat(out_list, axis=1)
            out = self.channel_shuffle(out, 2, self.out_planes)
            return out
            

class CatBottleneck_split(nn.Layer):
    def __init__(self, in_planes, out_planes, block_num=4, stride=1):
        super().__init__()
        assert block_num == 4, "block number should be larger than 1."
        assert stride in (1, 2)

        self.out_planes = out_planes
        self.stride = stride
        self.conv_list = nn.LayerList()

        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias_attr=False),
                nn.BatchNorm2D(out_planes // 2),
            )
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)

            for idx in range(block_num):
                if idx == 0:
                    self.conv_list.append(
                        ConvBNRelu(in_planes, out_planes // 2, kernel=1))
                elif idx < block_num - 1:
                    self.conv_list.append(
                        ConvBNRelu(out_planes // int(math.pow(2, idx)),
                                out_planes // int(math.pow(2, idx + 1))))
                else:
                    self.conv_list.append(
                        ConvBNRelu(out_planes // int(math.pow(2, idx)),
                                out_planes // int(math.pow(2, idx))))
        else:
            assert in_planes == out_planes
            self.conv_list.append(
                ConvBNRelu(in_planes // 2, in_planes // 4))
            self.conv_list.append(
                ConvBNRelu(in_planes // 4, in_planes // 8))
            self.conv_list.append(
                ConvBNRelu(in_planes // 8, in_planes // 8))

    def forward(self, x):
        if self.stride == 2:
            out_list = []
            out1 = self.conv_list[0](x)
            for idx, conv in enumerate(self.conv_list[1:]):
                if idx == 0:
                    if self.stride == 2:
                        out = conv(self.avd_layer(out1))
                    else:
                        out = conv(out1)
                else:
                    out = conv(out)
                out_list.append(out)

            if self.stride == 2:
                out1 = self.skip(out1)
            out_list.insert(0, out1)
            out = paddle.concat(out_list, axis=1)
            return out
        else:
            x1, x2 = paddle.split(x, num_or_sections=2, axis=1)
            out_list = []
            for conv in self.conv_list:
                x1 = conv(x1)
                out_list.append(x1)
            out_list.append(x2)
            out = paddle.concat(out_list, axis=1)
            return out


class CatBottleneck_dw(nn.Layer):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super().__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.LayerList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias_attr=False),
                nn.BatchNorm2D(out_planes // 2),
            )
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(in_planes, out_planes // 2, kernel=1))
            elif idx == 1:
                conv = nn.Sequential(
                    ConvBNRelu(out_planes // 2, out_planes // 4, groups=out_planes // 4),
                    ConvBNRelu(out_planes // 4, out_planes // 4, kernel=1))
                self.conv_list.append(conv)
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx))))

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        out = paddle.concat(out_list, axis=1)
        return out

class CatBottleneck_dw_pool(nn.Layer):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super().__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.LayerList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias_attr=False),
                nn.BatchNorm2D(out_planes // 2),
            )
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(in_planes, out_planes // 2, kernel=1))
            elif idx == 1:
                conv = nn.Sequential(
                    ConvBNRelu(out_planes // 2, out_planes // 4, groups=out_planes // 4),
                    ConvBNRelu(out_planes // 4, out_planes // 4, kernel=1))
                self.conv_list.append(conv)
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx + 1))))
            else:
                pool = paddle.nn.AvgPool2D(kernel_size=3, stride=1, padding=1)
                self.conv_list.append(pool)

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        out = paddle.concat(out_list, axis=1)
        return out

class CatBottleneck_dw_dilation_pool(nn.Layer):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super().__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.LayerList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias_attr=False),
                nn.BatchNorm2D(out_planes // 2),
            )
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(in_planes, out_planes // 2, kernel=1))
            elif idx == 1:
                conv = nn.Sequential(
                    ConvBNRelu(out_planes // 2, out_planes // 4, groups=out_planes // 4),
                    ConvBNRelu(out_planes // 4, out_planes // 4, kernel=1))
                self.conv_list.append(conv)
            elif idx == 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 4, out_planes // 8, dilation=1))
            else:
                pool = paddle.nn.AvgPool2D(kernel_size=3, stride=1, padding=1)
                self.conv_list.append(pool)

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        out = paddle.concat(out_list, axis=1)
        return out

'''
class CatBottleneck_repvgg(nn.Layer):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super().__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.LayerList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias_attr=False),
                nn.BatchNorm2D(out_planes // 2),
            )
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(
                    RepVGGBlock(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(
                    RepVGGBlock(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    RepVGGBlock(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(
                    RepVGGBlock(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx))))

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        out = paddle.concat(out_list, axis=1)
        return out
'''


class CatBottleneck_last_pool(nn.Layer):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super().__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.LayerList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias_attr=False),
                nn.BatchNorm2D(out_planes // 2),
            )
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx + 1))))
            else:
                pool = paddle.nn.AvgPool2D(kernel_size=5, stride=1, padding=2)
                self.conv_list.append(pool)

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        out = paddle.concat(out_list, axis=1)
        return out

class CatBottleneck_dilation_pool(nn.Layer):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super().__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.LayerList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias_attr=False),
                nn.BatchNorm2D(out_planes // 2),
            )
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(in_planes, out_planes // 2, kernel=1))
            elif idx == 1:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 4, stride=stride))
            elif idx == 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 4, out_planes // 8, dilation=2))
            else:
                pool = paddle.nn.AvgPool2D(kernel_size=3, stride=1, padding=1)
                self.conv_list.append(pool)

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        out = paddle.concat(out_list, axis=1)
        return out

class CatBottleneck_last_se(nn.Layer):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super().__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.LayerList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias_attr=False),
                nn.BatchNorm2D(out_planes // 2),
            )
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx))))
        self.conv_atten = ConvBNRelu(out_planes, out_planes, kernel=1, act=nn.Sigmoid)

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        out = paddle.concat(out_list, axis=1)

        atten = F.adaptive_avg_pool2d(out, 1)
        atten = self.conv_atten(atten)
        out = paddle.multiply(out, atten)

        return out
"""


class STDCNet(nn.Layer):
    '''
    The STDCNet implementation based on PaddlePaddle.

    The original article refers to Meituan
    Fan, Mingyuan, et al. "Rethinking BiSeNet For Real-time Semantic Segmentation."
    (https://arxiv.org/abs/2104.13188)

    Args:
        base(int, optional): base channels. Default: 64.
        layers(list, optional): layers numbers list. It determines STDC block numbers of STDCNet's stage3\4\5. Defualt: [4, 5, 3].
        block_num(int,optional): block_num of features block. Default: 4.
        type(str,optional): feature fusion method "cat"/"add". Default: "cat".
        num_classes(int, optional): class number for image classification. Default: 1000.
        dropout(float,optional): dropout ratio. if >0,use dropout ratio.  Default: 0.20.
        use_conv_last(bool,optional): whether to use the last ConvBNReLU layer . Default: False.
        pretrained(str, optional): the path of pretrained model.
    '''

    def __init__(self,
                 base=64,
                 layers=[4, 5, 3],
                 block_num=4,
                 type="cat",
                 num_classes=1000,
                 dropout=0.20,
                 use_conv_last=False,
                 pretrained=None):
        super(STDCNet, self).__init__()
        if type == "cat":
            block = CatBottleneck
        elif type == "add":
            block = AddBottleneck
        self.use_conv_last = use_conv_last
        self.features = self._make_layers(base, layers, block_num, block)
        self.conv_last = ConvBNRelu(base * 16, max(1024, base * 16), 1, 1)

        if (layers == [4, 5, 3]):  #stdc1446
            self.x2 = nn.Sequential(self.features[:1])
            self.x4 = nn.Sequential(self.features[1:2])
            self.x8 = nn.Sequential(self.features[2:6])
            self.x16 = nn.Sequential(self.features[6:11])
            self.x32 = nn.Sequential(self.features[11:])
        elif (layers == [2, 2, 2]):  #stdc813
            self.x2 = nn.Sequential(self.features[:1])
            self.x4 = nn.Sequential(self.features[1:2])
            self.x8 = nn.Sequential(self.features[2:4])
            self.x16 = nn.Sequential(self.features[4:6])
            self.x32 = nn.Sequential(self.features[6:])
        else:
            raise NotImplementedError(
                "model with layers:{} is not implemented!".format(layers))

        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        feat2 = self.x2(x)
        feat4 = self.x4(feat2)
        feat8 = self.x8(feat4)
        feat16 = self.x16(feat8)
        feat32 = self.x32(feat16)
        if self.use_conv_last:
            feat32 = self.conv_last(feat32)
        return feat2, feat4, feat8, feat16, feat32

    def _make_layers(self, base, layers, block_num, block):
        features = []
        features += [ConvBNRelu(3, base // 2, 3, 2)]
        features += [ConvBNRelu(base // 2, base, 3, 2)]

        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base * 4, block_num, 2))
                elif j == 0:
                    features.append(
                        block(base * int(math.pow(2, i + 1)),
                              base * int(math.pow(2, i + 2)), block_num, 2))
                else:
                    features.append(
                        block(base * int(math.pow(2, i + 2)),
                              base * int(math.pow(2, i + 2)), block_num, 1))

        return nn.Sequential(*features)

    def init_weight(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                param_init.normal_init(layer.weight, std=0.001)
            elif isinstance(layer, (nn.BatchNorm, nn.SyncBatchNorm)):
                param_init.constant_init(layer.weight, value=1.0)
                param_init.constant_init(layer.bias, value=0.0)
        if self.pretrained is not None:
            utils.load_pretrained_model(self, self.pretrained)


class ConvBNRelu(nn.Layer):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel=3,
                 stride=1,
                 dilation=1,
                 act=nn.ReLU,
                 groups=1):
        super(ConvBNRelu, self).__init__()
        self.conv = nn.Conv2D(
            in_planes,
            out_planes,
            kernel_size=kernel,
            stride=stride,
            padding=kernel // 2 if dilation == 1 else dilation,
            dilation=dilation,
            bias_attr=False,
            groups=groups)
        self.bn = nn.SyncBatchNorm(out_planes, data_format='NCHW')
        self.use_act = True if act is not None else False
        if self.use_act:
            self.act = act()

    def forward(self, x):
        out = self.bn(self.conv(x))
        if self.use_act:
            out = self.act(out)
        return out


class AddBottleneck(nn.Layer):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(AddBottleneck, self).__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.LayerList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias_attr=False),
                nn.BatchNorm2D(out_planes // 2),
            )
            self.skip = nn.Sequential(
                nn.Conv2D(
                    in_planes,
                    in_planes,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=in_planes,
                    bias_attr=False),
                nn.BatchNorm2D(in_planes),
                nn.Conv2D(
                    in_planes, out_planes, kernel_size=1, bias_attr=False),
                nn.BatchNorm2D(out_planes),
            )
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx))))

    def forward(self, x):
        out_list = []
        out = x
        for idx, conv in enumerate(self.conv_list):
            if idx == 0 and self.stride == 2:
                out = self.avd_layer(conv(out))
            else:
                out = conv(out)
            out_list.append(out)
        if self.stride == 2:
            x = self.skip(x)
        return paddle.concat(out_list, axis=1) + x


class CatBottleneck(nn.Layer):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(CatBottleneck, self).__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.LayerList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias_attr=False),
                nn.BatchNorm2D(out_planes // 2),
            )
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx))))

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        out = paddle.concat(out_list, axis=1)
        return out


class CatBottleneck_avg_4_block(nn.Layer):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super().__init__()
        assert block_num in (2, 4)
        assert stride in (1, 2)

        self.conv_list = nn.LayerList()
        self.stride = stride
        if stride == 2:
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)

        mid_chs = out_planes // block_num
        if block_num == 2:
            self.conv_list.append(
                ConvBNRelu(in_planes, mid_chs, kernel=2, stride=1))
            self.conv_list.append(
                ConvBNRelu(mid_chs, mid_chs, kernel=3, stride=stride))
        elif block_num == 4:
            self.conv_list.append(
                ConvBNRelu(in_planes, mid_chs, kernel=3, stride=1))
            self.conv_list.append(
                ConvBNRelu(mid_chs, mid_chs, kernel=3, stride=stride))
            self.conv_list.append(
                ConvBNRelu(mid_chs, mid_chs, kernel=3, stride=1))
            self.conv_list.append(
                ConvBNRelu(mid_chs, mid_chs, kernel=3, stride=1))

    def forward(self, x):
        out_list = []

        out = self.conv_list[0](x)
        if self.stride == 1:
            out_list.insert(0, out)
        else:
            out1 = self.skip(out)
            out_list.insert(0, out1)

        for _, conv in enumerate(self.conv_list[1:]):
            out = conv(out)
            out_list.append(out)

        out = paddle.concat(out_list, axis=1)
        return out


class CatBottleneck_add_short_cut(nn.Layer):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super().__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.LayerList()
        self.stride = stride

        self.short_cut = False
        if in_planes == out_planes and stride == 1:
            self.short_cut = True

        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias_attr=False),
                nn.BatchNorm2D(out_planes // 2),
            )
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx))))

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        out = paddle.concat(out_list, axis=1)
        if self.short_cut:
            out = out + x
        return out


class CatBottleneck_remove_dwconv(nn.Layer):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super().__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.LayerList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias_attr=False),
                nn.BatchNorm2D(out_planes // 2),
            )
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx))))

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        out = paddle.concat(out_list, axis=1)
        return out


class CatBottleneck_new_act(nn.Layer):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super().__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.LayerList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias_attr=False),
                nn.BatchNorm2D(out_planes // 2),
            )
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)
            stride = 1

        act = nn.Hardswish
        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(in_planes, out_planes // 2, kernel=1, act=act))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(
                    ConvBNRelu(
                        out_planes // 2,
                        out_planes // 2,
                        stride=stride,
                        act=act))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(
                    ConvBNRelu(
                        out_planes // 2,
                        out_planes // 4,
                        stride=stride,
                        act=act))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(
                        out_planes // int(math.pow(2, idx)),
                        out_planes // int(math.pow(2, idx + 1)),
                        act=act))
            else:
                self.conv_list.append(
                    ConvBNRelu(
                        out_planes // int(math.pow(2, idx)),
                        out_planes // int(math.pow(2, idx)),
                        act=act))

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        out = paddle.concat(out_list, axis=1)
        return out


class CatBottleneck_dilation_1(nn.Layer):
    '''the last one is dilation conv'''

    def __init__(self, in_planes, out_planes, block_num, stride=1):
        super().__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.LayerList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias_attr=False),
                nn.BatchNorm2D(out_planes // 2),
            )
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(
                    ConvBNRelu(
                        out_planes // int(math.pow(2, idx)),
                        out_planes // int(math.pow(2, idx)),
                        dilation=2))

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        out = paddle.concat(out_list, axis=1)
        return out


class CatBottleneck_dilation_2(nn.Layer):
    '''the last two is dilation conv'''

    def __init__(self, in_planes, out_planes, block_num, stride=1):
        super().__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.LayerList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias_attr=False),
                nn.BatchNorm2D(out_planes // 2),
            )
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 4, stride=stride))
            elif idx == block_num - 2:
                self.conv_list.append(
                    ConvBNRelu(
                        out_planes // int(math.pow(2, idx)),
                        out_planes // int(math.pow(2, idx + 1)),
                        dilation=2))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(
                    ConvBNRelu(
                        out_planes // int(math.pow(2, idx)),
                        out_planes // int(math.pow(2, idx)),
                        dilation=2))

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        out = paddle.concat(out_list, axis=1)
        return out


class CatBottleneck_gap_no_conv(nn.Layer):
    '''add global avg pooling in the last'''

    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super().__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.LayerList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias_attr=False),
                nn.BatchNorm2D(out_planes // 2),
            )
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx))))

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)

            if idx == len(self.conv_list[1:]) - 1:
                avg = F.adaptive_avg_pool2d(out, 1)
                out = out + avg
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        out = paddle.concat(out_list, axis=1)
        return out


class CatBottleneck_gap_conv(nn.Layer):
    '''add global avg pooling in the last'''

    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super().__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.LayerList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias_attr=False),
                nn.BatchNorm2D(out_planes // 2),
            )
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx))))
        ch = out_planes // int(math.pow(2, idx))
        self.conv = ConvBNRelu(ch, ch, kernel=1)

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)

            if idx == len(self.conv_list[1:]) - 1:
                avg = F.adaptive_avg_pool2d(out, 1)
                avg = self.conv(avg)
                out = out + avg
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        out = paddle.concat(out_list, axis=1)
        return out


class CatBottleneck_shuffle(nn.Layer):
    def __init__(self, in_planes, out_planes, block_num=4, stride=1):
        super().__init__()
        assert block_num == 4, "block number should be larger than 1."
        assert stride in (1, 2)

        self.out_planes = out_planes
        self.stride = stride
        self.conv_list = nn.LayerList()

        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias_attr=False),
                nn.BatchNorm2D(out_planes // 2),
            )
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)

            for idx in range(block_num):
                if idx == 0:
                    self.conv_list.append(
                        ConvBNRelu(in_planes, out_planes // 2, kernel=1))
                elif idx < block_num - 1:
                    self.conv_list.append(
                        ConvBNRelu(out_planes // int(math.pow(2, idx)),
                                   out_planes // int(math.pow(2, idx + 1))))
                else:
                    self.conv_list.append(
                        ConvBNRelu(out_planes // int(math.pow(2, idx)),
                                   out_planes // int(math.pow(2, idx))))
        else:
            assert in_planes == out_planes
            self.conv_list.append(ConvBNRelu(in_planes // 2, in_planes // 4))
            self.conv_list.append(ConvBNRelu(in_planes // 4, in_planes // 8))
            self.conv_list.append(ConvBNRelu(in_planes // 8, in_planes // 8))

    @staticmethod
    def channel_shuffle(x, groups, out_channels=None):
        x_shape = paddle.shape(x)
        if out_channels is None:
            out_channels = x_shape[1]
        channels_per_group = x_shape[1] // groups
        x = paddle.reshape(
            x=x,
            shape=[
                x_shape[0], groups, channels_per_group, x_shape[2], x_shape[3]
            ])
        x = paddle.transpose(x=x, perm=[0, 2, 1, 3, 4])
        x = paddle.flatten(x, start_axis=1, stop_axis=2)
        x = paddle.reshape(x=x, shape=[0, out_channels, 0, 0])
        return x

    def forward(self, x):
        if self.stride == 2:
            out_list = []
            out1 = self.conv_list[0](x)
            for idx, conv in enumerate(self.conv_list[1:]):
                if idx == 0:
                    if self.stride == 2:
                        out = conv(self.avd_layer(out1))
                    else:
                        out = conv(out1)
                else:
                    out = conv(out)
                out_list.append(out)

            if self.stride == 2:
                out1 = self.skip(out1)
            out_list.insert(0, out1)
            out = paddle.concat(out_list, axis=1)
            return out
        else:
            x1, x2 = paddle.split(x, num_or_sections=2, axis=1)
            out_list = []
            for conv in self.conv_list:
                x1 = conv(x1)
                out_list.append(x1)
            out_list.append(x2)
            out = paddle.concat(out_list, axis=1)
            out = self.channel_shuffle(out, 2, self.out_planes)
            return out


class CatBottleneck_shuffle_simple(nn.Layer):
    def __init__(self, in_planes, out_planes, block_num=4, stride=1):
        super().__init__()
        assert block_num == 4, "block number should be larger than 1."
        assert (stride == 1 and in_planes == out_planes) \
            or (stride == 2 and 2 * in_planes == out_planes)

        self.out_planes = out_planes
        self.stride = stride
        self.conv_list = nn.LayerList()

        if stride == 1:
            self.conv_list.append(ConvBNRelu(in_planes // 2, in_planes // 4))
            self.conv_list.append(ConvBNRelu(in_planes // 4, in_planes // 8))
            self.conv_list.append(ConvBNRelu(in_planes // 8, in_planes // 8))
        else:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    in_planes,
                    in_planes,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=in_planes,
                    bias_attr=False),
                nn.BatchNorm2D(in_planes),
            )
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)

            self.conv_list.append(ConvBNRelu(in_planes, in_planes // 2))
            self.conv_list.append(ConvBNRelu(in_planes // 2, in_planes // 4))
            self.conv_list.append(ConvBNRelu(in_planes // 4, in_planes // 4))

    @staticmethod
    def channel_shuffle(x, groups, out_channels=None):
        x_shape = paddle.shape(x)
        if out_channels is None:
            out_channels = x_shape[1]
        channels_per_group = x_shape[1] // groups
        x = paddle.reshape(
            x=x,
            shape=[
                x_shape[0], groups, channels_per_group, x_shape[2], x_shape[3]
            ])
        x = paddle.transpose(x=x, perm=[0, 2, 1, 3, 4])
        x = paddle.flatten(x, start_axis=1, stop_axis=2)
        x = paddle.reshape(x=x, shape=[0, out_channels, 0, 0])
        return x

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = paddle.split(x, num_or_sections=2, axis=1)
            out_list = []
            for conv in self.conv_list:
                x1 = conv(x1)
                out_list.append(x1)
            out_list.append(x2)
            out = paddle.concat(out_list, axis=1)
            out = self.channel_shuffle(out, 2, self.out_planes)
            return out
        else:
            out_list = []
            out1 = self.conv_list[0](self.avd_layer(x))
            out_list.append(out1)
            for idx, conv in enumerate(self.conv_list[1:]):
                out1 = conv(out1)
                out_list.append(out1)

            out2 = self.skip(x)
            out_list.append(out2)
            out = paddle.concat(out_list, axis=1)
            out = self.channel_shuffle(out, 2, self.out_planes)
            return out


class CatBottleneck_split(nn.Layer):
    def __init__(self, in_planes, out_planes, block_num=4, stride=1):
        super().__init__()
        assert block_num == 4, "block number should be larger than 1."
        assert stride in (1, 2)

        self.out_planes = out_planes
        self.stride = stride
        self.conv_list = nn.LayerList()

        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias_attr=False),
                nn.BatchNorm2D(out_planes // 2),
            )
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)

            for idx in range(block_num):
                if idx == 0:
                    self.conv_list.append(
                        ConvBNRelu(in_planes, out_planes // 2, kernel=1))
                elif idx < block_num - 1:
                    self.conv_list.append(
                        ConvBNRelu(out_planes // int(math.pow(2, idx)),
                                   out_planes // int(math.pow(2, idx + 1))))
                else:
                    self.conv_list.append(
                        ConvBNRelu(out_planes // int(math.pow(2, idx)),
                                   out_planes // int(math.pow(2, idx))))
        else:
            assert in_planes == out_planes
            self.conv_list.append(ConvBNRelu(in_planes // 2, in_planes // 4))
            self.conv_list.append(ConvBNRelu(in_planes // 4, in_planes // 8))
            self.conv_list.append(ConvBNRelu(in_planes // 8, in_planes // 8))

    def forward(self, x):
        if self.stride == 2:
            out_list = []
            out1 = self.conv_list[0](x)
            for idx, conv in enumerate(self.conv_list[1:]):
                if idx == 0:
                    if self.stride == 2:
                        out = conv(self.avd_layer(out1))
                    else:
                        out = conv(out1)
                else:
                    out = conv(out)
                out_list.append(out)

            if self.stride == 2:
                out1 = self.skip(out1)
            out_list.insert(0, out1)
            out = paddle.concat(out_list, axis=1)
            return out
        else:
            x1, x2 = paddle.split(x, num_or_sections=2, axis=1)
            out_list = []
            for conv in self.conv_list:
                x1 = conv(x1)
                out_list.append(x1)
            out_list.append(x2)
            out = paddle.concat(out_list, axis=1)
            return out


class CatBottleneck_dw(nn.Layer):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super().__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.LayerList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias_attr=False),
                nn.BatchNorm2D(out_planes // 2),
            )
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(in_planes, out_planes // 2, kernel=1))
            elif idx == 1:
                conv = nn.Sequential(
                    ConvBNRelu(
                        out_planes // 2,
                        out_planes // 4,
                        groups=out_planes // 4),
                    ConvBNRelu(out_planes // 4, out_planes // 4, kernel=1))
                self.conv_list.append(conv)
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx))))

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        out = paddle.concat(out_list, axis=1)
        return out


class CatBottleneck_dw_pool3(nn.Layer):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super().__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.LayerList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias_attr=False),
                nn.BatchNorm2D(out_planes // 2),
            )
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(in_planes, out_planes // 2, kernel=1))
            elif idx == 1:
                conv = nn.Sequential(
                    ConvBNRelu(
                        out_planes // 2,
                        out_planes // 4,
                        groups=out_planes // 4),
                    ConvBNRelu(out_planes // 4, out_planes // 4, kernel=1))
                self.conv_list.append(conv)
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx + 1))))
            else:
                pool = paddle.nn.AvgPool2D(kernel_size=3, stride=1, padding=1)
                self.conv_list.append(pool)

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        out = paddle.concat(out_list, axis=1)
        return out


class CatBottleneck_dw_pool5(nn.Layer):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super().__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.LayerList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias_attr=False),
                nn.BatchNorm2D(out_planes // 2),
            )
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(in_planes, out_planes // 2, kernel=1))
            elif idx == 1:
                conv = nn.Sequential(
                    ConvBNRelu(
                        out_planes // 2,
                        out_planes // 4,
                        groups=out_planes // 4),
                    ConvBNRelu(out_planes // 4, out_planes // 4, kernel=1))
                self.conv_list.append(conv)
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx + 1))))
            else:
                pool = paddle.nn.AvgPool2D(kernel_size=5, stride=1, padding=2)
                self.conv_list.append(pool)

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        out = paddle.concat(out_list, axis=1)
        return out


class CatBottleneck_dw_pool7(nn.Layer):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super().__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.LayerList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias_attr=False),
                nn.BatchNorm2D(out_planes // 2),
            )
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(in_planes, out_planes // 2, kernel=1))
            elif idx == 1:
                conv = nn.Sequential(
                    ConvBNRelu(
                        out_planes // 2,
                        out_planes // 4,
                        groups=out_planes // 4),
                    ConvBNRelu(out_planes // 4, out_planes // 4, kernel=1))
                self.conv_list.append(conv)
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx + 1))))
            else:
                pool = paddle.nn.AvgPool2D(kernel_size=7, stride=1, padding=3)
                self.conv_list.append(pool)

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        out = paddle.concat(out_list, axis=1)
        return out


class CatBottleneck_dw_dilation_pool(nn.Layer):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super().__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.LayerList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias_attr=False),
                nn.BatchNorm2D(out_planes // 2),
            )
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(in_planes, out_planes // 2, kernel=1))
            elif idx == 1:
                conv = nn.Sequential(
                    ConvBNRelu(
                        out_planes // 2,
                        out_planes // 4,
                        groups=out_planes // 4),
                    ConvBNRelu(out_planes // 4, out_planes // 4, kernel=1))
                self.conv_list.append(conv)
            elif idx == 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 4, out_planes // 8, dilation=2))
            else:
                pool = paddle.nn.AvgPool2D(kernel_size=3, stride=1, padding=1)
                self.conv_list.append(pool)

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        out = paddle.concat(out_list, axis=1)
        return out


class CatBottleneck_dw_dilation(nn.Layer):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super().__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.LayerList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias_attr=False),
                nn.BatchNorm2D(out_planes // 2),
            )
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(in_planes, out_planes // 2, kernel=1))
            elif idx == 1:
                conv = nn.Sequential(
                    ConvBNRelu(
                        out_planes // 2,
                        out_planes // 4,
                        groups=out_planes // 4),
                    ConvBNRelu(out_planes // 4, out_planes // 4, kernel=1))
                self.conv_list.append(conv)
            elif idx == 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 4, out_planes // 8))
            else:
                self.conv_list.append(
                    ConvBNRelu(
                        out_planes // int(math.pow(2, idx)),
                        out_planes // int(math.pow(2, idx)),
                        dilation=2))

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        out = paddle.concat(out_list, axis=1)
        return out


class CatBottleneck_last_pool(nn.Layer):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super().__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.LayerList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias_attr=False),
                nn.BatchNorm2D(out_planes // 2),
            )
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx + 1))))
            else:
                pool = paddle.nn.AvgPool2D(kernel_size=5, stride=1, padding=2)
                self.conv_list.append(pool)

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        out = paddle.concat(out_list, axis=1)
        return out


class CatBottleneck_dilation_pool(nn.Layer):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super().__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.LayerList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias_attr=False),
                nn.BatchNorm2D(out_planes // 2),
            )
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(in_planes, out_planes // 2, kernel=1))
            elif idx == 1:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 4, stride=stride))
            elif idx == 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 4, out_planes // 8, dilation=2))
            else:
                pool = paddle.nn.AvgPool2D(kernel_size=3, stride=1, padding=1)
                self.conv_list.append(pool)

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        out = paddle.concat(out_list, axis=1)
        return out


class CatBottleneck_last_se(nn.Layer):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super().__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.LayerList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias_attr=False),
                nn.BatchNorm2D(out_planes // 2),
            )
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx))))
        self.conv_atten = ConvBNRelu(
            out_planes, out_planes, kernel=1, act=nn.Sigmoid)

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        out = paddle.concat(out_list, axis=1)

        atten = F.adaptive_avg_pool2d(out, 1)
        atten = self.conv_atten(atten)
        out = paddle.multiply(out, atten)

        return out


class CatBottleneck_conv5(nn.Layer):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super().__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.LayerList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias_attr=False),
                nn.BatchNorm2D(out_planes // 2),
            )
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)

        self.conv_list.append(ConvBNRelu(in_planes, out_planes // 2, kernel=1))
        self.conv_list.append(ConvBNRelu(out_planes // 2, out_planes // 4))
        self.conv_list.append(ConvBNRelu(out_planes // 4, out_planes // 8))
        self.conv_list.append(
            ConvBNRelu(out_planes // 8, out_planes // 8, kernel=5))

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        out = paddle.concat(out_list, axis=1)
        return out


class ConvBN(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups=1):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias_attr=False)
        self.bn = nn.BatchNorm2D(num_features=out_channels)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        return y


class RepVGGBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros'):
        super(RepVGGBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        self.rbr_identity = nn.BatchNorm2D(
            num_features=in_channels
        ) if out_channels == in_channels and stride == 1 else None
        self.rbr_dense = ConvBN(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups)
        self.rbr_1x1 = ConvBN(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding_11,
            groups=groups)

    def forward(self, inputs):
        if not self.training:
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        return self.nonlinearity(
            self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def eval(self):
        if not hasattr(self, 'rbr_reparam'):
            self.rbr_reparam = nn.Conv2D(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                padding_mode=self.padding_mode)
        self.training = False
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam.weight.set_value(kernel)
        self.rbr_reparam.bias.set_value(bias)
        for layer in self.sublayers():
            layer.eval()

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(
            kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, ConvBN):
            kernel = branch.conv.weight
            running_mean = branch.bn._mean
            running_var = branch.bn._variance
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn._epsilon
        else:
            assert isinstance(branch, nn.BatchNorm2D)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = paddle.to_tensor(kernel_value)
            kernel = self.id_tensor
            running_mean = branch._mean
            running_var = branch._variance
            gamma = branch.weight
            beta = branch.bias
            eps = branch._epsilon
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))
        return kernel * t, beta - running_mean * gamma / std

class CatBottleneck_repvgg(nn.Layer):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super().__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.LayerList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias_attr=False),
                nn.BatchNorm2D(out_planes // 2),
            )
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(
                    RepVGGBlock(out_planes // 2, out_planes // 2))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(
                    RepVGGBlock(out_planes // 2, out_planes // 4))
            elif idx < block_num - 1:
                self.conv_list.append(
                    RepVGGBlock(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(
                    RepVGGBlock(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx))))

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        out = paddle.concat(out_list, axis=1)
        return out

class AttenNone(nn.Layer):
    def __init__(self, in_ch):
        super().__init__()

    def forward(self, x):
        return x

class AttenSE_conv(nn.Layer):
    def __init__(self, ch, ratio=16):
        super().__init__()
        self.fc1 = nn.Conv2D(ch, ch//ratio, 1, bias_attr=False)
        self.fc2 = nn.Conv2D(ch//ratio, ch, 1, bias_attr=False)

    def forward(self, x):
        atten = F.adaptive_avg_pool2d(x, 1)
        atten = F.relu(self.fc1(atten))
        atten = F.sigmoid(self.fc2(atten))
        out = atten * x
        return out

class AttenSE_linear(nn.Layer):
    def __init__(self, ch, ratio=16):
        super().__init__()
        self.fc1 = nn.Linear(ch, ch//ratio, bias_attr=False)
        self.fc2 = nn.Linear(ch//ratio, ch, bias_attr=False)

    def forward(self, x):
        atten = F.adaptive_avg_pool2d(x, 1)
        atten = paddle.squeeze(atten, axis=(2,3))
        atten = F.relu(self.fc1(atten))
        atten = F.sigmoid(self.fc2(atten))
        atten = paddle.unsqueeze(atten, axis=(2,3))
        out = atten * x
        return out

class AttenSESimple_1(nn.Layer):
    def __init__(self, ch):
        super().__init__()
        self.fc = nn.Conv2D(ch, ch, 1, bias_attr=False)

    def forward(self, x):
        atten = F.adaptive_avg_pool2d(x, 1)
        atten = F.sigmoid(self.fc(atten))
        out = x + atten * x
        return out

class AttenSESimple_2(nn.Layer):
    def __init__(self, ch):
        super().__init__()
        self.fc = nn.Conv2D(ch, ch, 1, bias_attr=False)

    def forward(self, x):
        atten = F.adaptive_avg_pool2d(x, 1)
        atten = F.sigmoid(self.fc(atten))
        out = atten * x
        return out

class STDCNet_pp_1(nn.Layer):
    '''support all block type'''
    def __init__(self,
                 base=64,
                 layers=[2, 2, 2],
                 layers_expand=[4, 8, 16],
                 block_num=4,
                 block_type="CatBottleneck",
                 num_classes=1000,
                 dropout=0.20,
                 atten_type='AttenNone',
                 pretrained=None):
        super().__init__()

        print("block_type:" + block_type)
        block = eval(block_type)
        self.feat_channels = [v * base for v in layers_expand]

        self.features = self._make_layers(base, layers, layers_expand, block_num, block)

        self.conv_last = ConvBNRelu(self.feat_channels[-1], 1024, 1, 1)
        self.fc = nn.Linear(1024, 1024, bias_attr=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(1024, num_classes, bias_attr=False)

        self.x2 = nn.Sequential(self.features[:1])
        self.x4 = nn.Sequential(self.features[1:2])
        idx = (2 + layers[0], 2 + layers[0] + layers[1])
        self.x8 = nn.Sequential(self.features[2:idx[0]])
        self.x16 = nn.Sequential(self.features[idx[0]:idx[1]])
        self.x32 = nn.Sequential(self.features[idx[1]:])

        print("atten_type:" + atten_type)
        atten = eval(atten_type)
        self.f4_atten = atten(base)
        self.f8_atten = atten(self.feat_channels[0])
        self.f16_atten = atten(self.feat_channels[1])
        self.f32_atten = atten(self.feat_channels[2])

        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        feat2 = self.x2(x)

        feat4 = self.x4(feat2)
        feat4 = self.f4_atten(feat4)
        
        feat8 = self.x8(feat4)
        feat8 = self.f8_atten(feat8)
        
        feat16 = self.x16(feat8)
        feat16 = self.f16_atten(feat16)
        
        feat32 = self.x32(feat16)
        feat32 = self.f32_atten(feat32)

        out = self.conv_last(feat32).pow(2)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.flatten(1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear(out)
        return out

    def _make_layers(self, base, layers, layers_expand, block_num, block):
        assert len(layers) == 3 and len(layers_expand) == 3

        features = []
        features += [ConvBNRelu(3, base // 2, 3, 2)]
        features += [ConvBNRelu(base // 2, base, 3, 2)]

        l0, l1, l2 = layers[0], layers[1], layers[2]
        r0, r1, r2 = layers_expand[0], layers_expand[1], layers_expand[2]

        features.append(block(base, base * r0, block_num, 2))
        for i in range(l0 - 1):
            features.append(block(base * r0, base * r0, block_num, 1))

        features.append(block(base * r0, base * r1, block_num, 2))
        for i in range(l1 - 1):
            features.append(block(base * r1, base * r1, block_num, 1))

        features.append(block(base * r1, base * r2, block_num, 2))
        for i in range(l2 - 1):
            features.append(block(base * r2, base * r2, block_num, 1))

        return nn.Sequential(*features)

    def init_weight(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                param_init.normal_init(layer.weight, std=0.001)
            elif isinstance(layer, (nn.BatchNorm, nn.SyncBatchNorm)):
                param_init.constant_init(layer.weight, value=1.0)
                param_init.constant_init(layer.bias, value=0.0)
        if self.pretrained is not None:
            utils.load_pretrained_model(self, self.pretrained)



def _load_pretrained(pretrained, model, model_url, use_ssld=False):
    if pretrained is False:
        pass
    elif pretrained is True:
        load_dygraph_pretrain_from_url(model, model_url, use_ssld=use_ssld)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )

def STDCNet_1(pretrained=False, use_ssld=False, **kwargs):
    model = STDCNet_pp_1(base=64, layers=[2, 2, 2], layers_expand=[4, 8, 16], **kwargs)
    _load_pretrained(
        pretrained, model, "", use_ssld=use_ssld)
    return model

def STDCNet_2(pretrained=False, use_ssld=False, **kwargs):
    model = STDCNet_pp_1(base=64, layers=[4, 5, 3], **kwargs)
    _load_pretrained(
        pretrained, model, "", use_ssld=use_ssld)
    return model

def STDCNet_3(pretrained=False, use_ssld=False, **kwargs):
    model = STDCNet_pp_1(base=64, layers=[2, 2, 2], layers_expand=[4, 8, 8], **kwargs)
    _load_pretrained(
        pretrained, model, "", use_ssld=use_ssld)
    return model