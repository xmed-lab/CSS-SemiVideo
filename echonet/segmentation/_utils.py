from collections import OrderedDict
from typing import Optional, Dict

from torch import nn, Tensor
from torch.nn import functional as F
import torch 


class _SimpleSegmentationModel(nn.Module):
    __constants__ = ['aux_classifier']

    def __init__(
        self,
        backbone: nn.Module,
        classifier: nn.Module,
        aux_classifier: Optional[nn.Module] = None
    ) -> None:
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        # self.aux_classifier = aux_classifier

        self.ctr_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.ctr_fc = nn.Sequential(nn.Linear(2048, 2048), nn.ReLU(), nn.Linear(2048, 128))


    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        
        # print("self.backbone", self.backbone.conv1)
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = x

        x_ctr = features["out"]
        x_ctr = self.ctr_avgpool(x_ctr)
        x_ctr = x_ctr.flatten(1)
        x_ctr = self.ctr_fc(x_ctr)
        # print("x_ctr.shape, in _utils segmentation", x_ctr.shape)
        result['ctr_feat'] = F.normalize(x_ctr, dim = 1)
        result['feat_mid'] = features["out"]
        return result





class _SimpleSegmentationModel_CSS(nn.Module):
    __constants__ = ['aux_classifier']

    def __init__(
        self,
        backbone: nn.Module,
        classifier: nn.Module,
        aux_classifier: Optional[nn.Module] = None
    ) -> None:
        super(_SimpleSegmentationModel_CSS, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        # self.aux_classifier = aux_classifier


    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        
        # print("self.backbone", self.backbone.conv1)

        xtest = self.backbone.conv1(x)
        xtest = self.backbone.bn1(xtest)
        xtest = self.backbone.relu(xtest)
        xtest_layerbs = xtest
        xtest = self.backbone.maxpool(xtest)
        xtest_layer0 = xtest
        xtest = self.backbone.layer1(xtest)
        xtest_layer1 = xtest
        xtest = self.backbone.layer2(xtest) ### can just output here. 
        xtest_layer2 = xtest
        xtest = self.backbone.layer3(xtest)
        xtest = self.backbone.layer4(xtest)
        # print("xtest_layerbs.shape", xtest_layerbs.shape)# xtest_layerbs.shape torch.Size([2, 64, 56, 56])
        # print("xtest_layer0.shape", xtest_layer0.shape) #xtest_layer0.shape torch.Size([2, 64, 28, 28])
        # print("xtest_layer1.shape", xtest_layer1.shape) #xtest_layer1.shape torch.Size([2, 256, 28, 28])
        # print("xtest_layer2.shape", xtest_layer2.shape) #torch.Size([2, 512, 14, 14])

        result = OrderedDict()
        x = xtest

        x = self.classifier(x)
        x_maskpre = x
        x_maskpre = F.interpolate(x_maskpre, size=[56,56], mode='bilinear', align_corners=False)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = x
        result['x_layerbs'] = xtest_layerbs
        result['x_layer1'] = xtest_layer1
        result['x_layer4'] = xtest
        result['maskfeat'] = x_maskpre
        return result


