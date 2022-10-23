# torch import
import torch,utils
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torchvision.models._utils import IntermediateLayerGetter
import pytorch_lightning as pl


class SMNet(pl.LightningModule):    
    def __init__(self,style:torch.Tensor,content_weight:int=1,style_weight:int=1e3,automatic_optimization=True) -> None:
        super().__init__()
        self.style,self.content_weight,self.style_weight = style,content_weight,style_weight
        self.vgg=models.vgg16(pretrained=True)
        self.input_image = nn.Parameter(torch.rand(style.size()).data)
        self.automatic_optimization = automatic_optimization
        self.feature_net = IntermediateLayerGetter(self.vgg.features,{
            '3':'layer1_2',
            '8':'layer2_2',
            '15':'layer3_3',
            '22':'layer4_3',
        })
        
    def training_step(self,batch,batch_index):
        opt = self.optimizers()
        opt.zero_grad()
        if(str(self.device).find('cuda') != -1 and str(self.style.device) != str(self.device)):
            self.style = self.style.to(self.device)
            self.input_image = self.input_image.to(self.device)
        style_features = self.feature_net(self.style)
        content_features = self.feature_net(batch)
        input_features = self.feature_net(self.input_image)
        content_loss = 0
        for layer in style_features:
            content_loss += F.mse_loss(
                input_features[layer], content_features[layer])
        content_loss = self.content_weight*content_loss
        style_loss = 0
        for layer in style_features:
            style_loss += F.mse_loss(utils.gram_matrix(style_features[layer].expand_as(input_features[layer])),utils.gram_matrix(input_features[layer]))
        style_loss = self.style_weight * style_loss
        loss = style_loss+content_loss
        self.manual_backward(loss,retain_graph = True)
        self.log('loss', loss, prog_bar=True)
        # print('input image grad: '+str(self.input_image.grad))
        opt.step()
        # loss.backward(retain_graph = True)
    
    def configure_optimizers(self):
        opt = torch.optim.Adam([self.input_image])
        return opt
        
        

