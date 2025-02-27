# Retinanet

Retinanet è un esempio di architettura di rete single stage. È costituita da un modulo principale chiamato *backbone* e da due moduli secondari chiamati *subnetwork*. Il primo estrae le feature map dall'intera immagine mentre le subnet costituiscono i moduli di classificazione e regressione.

La rete di Backbone è costituita dal Feature Pyramid Network (FPN). Essenzialmente vengono estratte delle feature map a differenti valori di scale con lo scopo di identificare oggetti di dimensione diversa.

La rete combina le feature *semanticamente forti* a bassa risoluzione con caratteristiche *semanticamente deboli* ad alta risoluzione tramite un Top-Down path e con connessioni laterali.

```python
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
```

```python
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class BasicBlockFeatures(BasicBlock):
    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        conv2_rep = out
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out, conv2_rep

class BottleneckFeatures(Bottleneck):
    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        conv3_rep = out
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out, conv3_rep

class ResNetFeatures(ResNet):
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x, c2 = self.layer1(x)
        x, c3 = self.layer2(x)
        x, c4 = self.layer3(x)
        x, c5 = self.layer4(x)

        return c2, c3, c4, c5

def resnet18(pretrained=False, **kwargs):
    model = ResNetFeatures(BasicBlockFeatures, [2, 2, 2, 2], **kwargs)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))

    return model

def resnet34(pretrained=False, **kwargs):
    model = ResNetFeatures(BasicBlockFeatures, [3, 4, 6, 3], **kwargs)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))

    return model

def resnet50(pretrained=False, **kwargs):
    model = ResNetFeatures(BottleneckFeatures, [3, 4, 6, 3], **kwargs)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))

    return model

def resnet101(pretrained=False, **kwargs):
    model = ResNetFeatures(BottleneckFeatures, [3, 4, 23, 3], **kwargs)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))

    return model

def resnet152(pretrained=False, **kwargs):
    model = ResNetFeatures(BottleneckFeatures, [3, 8, 36, 3], **kwargs)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))

    return model
```

```python
def classification_layer_init(tensor, pi=0.01):
    fill_constant = - math.log((1 - pi) / pi)
    if isinstance(tensor, Variable):
        classification_layer_init(tensor.data)
    return tensor.fill_(fill_constant)

def init_conv_weights(layer):
    nn.init.normal(layer.weight.data, std=0.01)
    nn.init.constant(layer.bias.data, val=0)
    return layer

def conv1x1(in_channels, out_channels, **kwargs):
    layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, **kwargs)
    layer = init_conv_weights(layer)
    return layer

def conv3x3(in_channels, out_channels, **kwargs):
    layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, **kwargs)
    layer = init_conv_weights(layer)
    return layer

def upsample(feature, sample_feature, scale_factor=2):
    h, w = sample_feature.size()[2:]
    return F.upsample(feature, scale_factor=scale_factor)[:, :, :h, :w]
```

```python
class FeaturePyramid(nn.Module):
    def __init__(self, resnet):
        super(FeaturePyramid, self).__init__()

        self.resnet = resnet

        self.pyramid_transformation_3 = conv1x1(512, 256)
        self.pyramid_transformation_4 = conv1x1(1024, 256)
        self.pyramid_transformation_5 = conv1x1(2048, 256)

        self.pyramid_transformation_6 = conv3x3(2048, 256, padding=1, stride=2)
        self.pyramid_transformation_7 = conv3x3(256, 256, padding=1, stride=2)

        self.upsample_transform_1 = conv3x3(256, 256, padding=1)
        self.upsample_transform_2 = conv3x3(256, 256, padding=1)

    def forward(self, x):
        _, resnet_feature_3, resnet_feature_4, resnet_feature_5 = self.resnet(x)

        pyramid_feature_6 = self.pyramid_transformation_6(resnet_feature_5)
        pyramid_feature_7 = self.pyramid_transformation_7(F.relu(pyramid_feature_6))

        pyramid_feature_5 = self.pyramid_transformation_5(resnet_feature_5)

        pyramid_feature_4 = self.pyramid_transformation_4(resnet_feature_4)
        upsampled_feature_5 = upsample(pyramid_feature_5, pyramid_feature_4)
        pyramid_feature_4 = self.upsample_transform_1(torch.add(upsampled_feature_5, pyramid_feature_4))

        pyramid_feature_3 = self.pyramid_transformation_3(resnet_feature_3)
        upsampled_feature_4 = upsample(pyramid_feature_4, pyramid_feature_3)
        pyramid_feature_3 = self.upsample_transform_2(torch.add(upsampled_feature_4, pyramid_feature_3))

        return pyramid_feature_3, pyramid_feature_4, pyramid_feature_5, pyramid_feature_6, pyramid_feature_7
```

```python
class SubNet(nn.Module):
    def __init__(self, k, anchors=9, depth=4, activation=F.relu):
        super(SubNet, self).__init__()
        self.anchors = anchors
        self.activation = activation
        self.base = nn.ModuleList([conv3x3(256, 256, padding=1) for _ in range(depth)])
        self.output = nn.Conv2d(256, k * anchors, kernel_size=3, padding=1)
        classification_layer_init(self.output.weight.data)

    def forward(self, x):
        for layer in self.base:
            x = self.activation(layer(x))
        x = self.output(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(x.size(0), x.size(2) * x.size(3) * self.anchors, -1)
        return x
```

```python
class RetinaNet(nn.Module):
    backbones = {
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'resnet101': resnet101,
        'resnet152': resnet152
    }

    def __init__(self, backbone='resnet101', num_classes=20, pretrained=True):
        super(RetinaNet, self).__init__()
        self.resnet = RetinaNet.backbones[backbone](pretrained=pretrained)
        self.feature_pyramid = FeaturePyramid(self.resnet)
        self.subnet_boxes = SubNet(4)
        self.subnet_classes = SubNet(num_classes + 1)

    def forward(self, x):
        pyramid_features = self.feature_pyramid(x)
        class_predictions = [self.subnet_classes(p) for p in pyramid_features]
        bbox_predictions = [self.subnet_boxes(p) for p in pyramid_features]
        return torch.cat(bbox_predictions, 1), torch.cat(class_predictions, 1)

```

# Loss

L'altra innovazione introdotta in RetinaNet è la formulazione della loss. Come gestire il problema delle classi sbilanciate?

Il numero degli anchor box negativi è molto maggiore dei box positivi. Soluzione: pesare opportunamente le predizioni corrette rispetto a quelle errate.

dove $\alpha_t$ è un paramtro di bilanciamento, $p_t$ è la probabilità associata alla classe *t*, $\gamma$ è definito come *focusing parameter*.
