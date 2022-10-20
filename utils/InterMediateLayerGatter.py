from collections import OrderedDict
from typing import Dict
import torch.nn as nn


class InterMediateLayerGatter(object):
    def __init__(self, model: nn.Module, layers: Dict) -> None:
        super().__init__()
        self.model = model
        self.layers = layers

    def __call__(self, x):
        res = OrderedDict()
        # this is the hook version
        def get_hook(layer_key):
            def get_output_hook(module, fea_in, fea_out):
                res[layer_key] = fea_out
            return get_output_hook
        for layer in self.layers.keys():
            net = self.model
            for i in layer.split('/'):
                net = getattr(net, i)
            net.register_forward_hook(hook=get_hook(self.layers[layer]))
        self.model(x)
        return res
