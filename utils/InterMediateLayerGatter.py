from collections import OrderedDict
from turtle import forward
from typing import Dict
import torch.nn as nn

class CatchLayer(nn.Module):
    def __init__(self,layer:nn.Module) -> None:
        super().__init__()
        self.layer = layer
    
    def forward(self,x):
        x = self.layer(x)
        self.y = x
        return x
        
    def get_catch(self):
        return self.y

class InterMediateLayerGatter(nn.Module):    
    def __init__(self, model: nn.Module, layers: Dict) -> None:
        super().__init__()
        self.model = model
        self.layers = layers

    def forward(self, x):
        res = OrderedDict()
        for name,layer in self.model.named_modules():
            if name in self.layers.keys():
                name_s = name.split('.')
                if(len(name_s)>1):
                    layer_p = getattr(self.model,'.'.join(name_s[:-1]))
                layer_p.add_module(name_s[-1],CatchLayer(layer))
        self.model(x)
        for name,layer in self.model.named_modules():
            if name in self.layers.keys():
                res[self.layers[name]] = layer.get_catch()
        return res
