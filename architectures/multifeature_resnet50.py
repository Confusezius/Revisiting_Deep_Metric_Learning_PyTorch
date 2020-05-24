"""
The network architectures and weights are adapted and used from the great https://github.com/Cadene/pretrained-models.pytorch.
"""
import torch, torch.nn as nn
import pretrainedmodels as ptm





"""============================================================="""
class Network(torch.nn.Module):
    def __init__(self, opt):
        super(Network, self).__init__()

        self.pars  = opt
        self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained='imagenet')

        self.name = opt.arch

        if 'frozen' in opt.arch:
            for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
                module.eval()
                module.train = lambda _: None

        self.feature_dim = self.model.last_linear.in_features
        out_dict = nn.ModuleDict()
        for mode in opt.diva_features:
            out_dict[mode] = torch.nn.Linear(self.feature_dim, opt.embed_dim)

        self.model.last_linear  = out_dict

        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])


    def forward(self, x):
        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))
        for layerblock in self.layer_blocks:
            x = layerblock(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0),-1)

        out_dict = {}
        for key,linear_map in self.model.last_linear.items():
            if not 'normalize' in self.pars.arch:
                out_dict[key] = linear_map(x)
            else:
                out_dict[key] = torch.nn.functional.normalize(linear_map(x), dim=-1)

        return out_dict, x
