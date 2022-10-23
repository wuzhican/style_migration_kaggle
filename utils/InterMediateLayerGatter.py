from collections import OrderedDict
from turtle import forward
from typing import Dict
import torch.nn as nn

class CatchLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self,x):
        self.x = x
        return x
        
    def get_catch(self):
        return self.x

class InterMediateLayerGatter(nn.Module):    
    def __init__(self, model: nn.Module, layers: Dict) -> None:
        super().__init__()
        self.model = model
        self.layers = layers
        
    @staticmethod
    def __get_sub_layer(module,layer_path):
        for l in layer_path.split('.'):
            module = getattr(module,l)
        return module

    def forward(self, x):
        res = OrderedDict()
        for name in self.layers.keys():
            name_s = name.split('.')
            layer = self.__get_sub_layer(self.model,name)
            layer_p = self.__get_sub_layer(self.model,'.'.join(name_s[:-1]))
            layer_p.add_module(name_s[-1],nn.Sequential(layer,CatchLayer()))
        self.model(x)
        for name in self.layers.keys():
            layer = self.__get_sub_layer(self.model,name)
            res[self.layers[name]]=layer[1].get_catch()
        return res
