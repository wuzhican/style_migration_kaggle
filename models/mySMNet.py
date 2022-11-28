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
    
    def __init__(self,style:torch.Tensor,**args) -> None:
        super().__init__()
        self.style = style
        args_v = {
            'content_weight': 1e3,
            'style_weight': 1,
            'automatic_optimization': True,
            'content_layers': ['layer1_2', 'layer2_2', 'layer3_3', 'layer4_3', 'layer5_3'],
            'style_layers': ['layer1_2', 'layer2_2', 'layer3_3', 'layer4_3', 'layer5_3'],
            'train_epochs': 50,
        }
        for key in args_v.keys():
            if key in args.keys():
                setattr(self,key,args[key])
            else:
                setattr(self,key,args_v[key])
        self.vgg = models.vgg16(pretrained=True)
        self.input_image = nn.Parameter(torch.rand(style.size()).data)
        self.feature_net = IntermediateLayerGetter(self.vgg.features, {
            '3':'layer1_2',
            '8':'layer2_2',
            '15':'layer3_3',
            '22':'layer4_3',
            '29':'layer5_3'
        })
        
    def training_step(self,batch,batch_index):
        opt = self.optimizers()
        opt.zero_grad()
        print('before trans batch device:%s'%(batch.device))
        if(str(self.device).find('cuda') != -1 and str(self.style.device) != str(self.device)):
            self.style = self.style.to(self.device)
            self.input_image = self.input_image.to(self.device)
        if(str(self.device).find('XLA') != -1 and str(self.style.device) != str(batch.device)):
            batch = batch.to(self.device)
            print('after trans batch device:%s'%(batch.device))
        self.input_image.data.clamp_(0,1)
        print('feature_net device:%s'%(self.feature_net.device))
        print('style device:%s'%(self.style.device))
        style_features = self.feature_net(self.style)
        content_features = self.feature_net(batch)
        print('input_image device:%s'%(self.input_image.device))
        input_features = self.feature_net(self.input_image)
        content_loss = 0
        for layer in style_features:
            if layer in self.content_layers:
                content_loss += F.mse_loss(
                    input_features[layer].expand_as(content_features[layer]), content_features[layer])
        content_loss = self.content_weight*content_loss
        style_loss = 0
        for layer in style_features:
            if layer in self.style_layers:
                style_loss += F.mse_loss(utils.gram_matrix(style_features[layer].expand_as(
                    input_features[layer])), utils.gram_matrix(input_features[layer]))
        style_loss = self.style_weight * style_loss
        loss = style_loss+content_loss
        self.manual_backward(loss,retain_graph = True)
        self.log('loss', loss, prog_bar=True)
        # print('input image grad: '+str(self.input_image.grad))
        opt.step()
        # loss.backward(retain_graph = True)
    
    def on_train_batch_end(self, outputs, batch, batch_idx: int, unused: int = 0) -> None:
        if batch_idx%self.train_epochs == self.train_epochs - 1:
            title = 'epoch %s'%(int((batch_idx+1)/self.train_epochs))
            # utils.show_tensor(self.input_image,title=title)
            utils.show_tensor(self.input_image,utils.show_image,title)
    
    def configure_optimizers(self):
        opt = torch.optim.Adam([self.input_image])
        return opt
        
        

