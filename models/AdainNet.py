import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models._utils import IntermediateLayerGetter

class VggConvLayer(nn.Module):
    def __init__(self,in_channel,out_channel,layer_num=1) -> None:
        super().__init__()
        assert layer_num >= 1
        modules = []
        for i in range(layer_num):
            modules.append(nn.ReflectionPad2d(1))
            modules.append(nn.Conv2d(in_channel,out_channel,3))
            modules.append(nn.ReLU())
        self.layers = nn.Sequential(*modules)

            
    def forward(self,x):
        return self.layers(x)
        
        
class VggEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3,3,1),
            VggConvLayer(3,64),
            VggConvLayer(64,64),
            nn.MaxPool2d(2,2,ceil_mode=True),
            VggConvLayer(64,128),
            VggConvLayer(128,128),
            nn.MaxPool2d(2,2,ceil_mode=True),
            VggConvLayer(128,256),
            VggConvLayer(256,256,3),
            nn.MaxPool2d(2,2,ceil_mode=True),
            VggConvLayer(256,512),
            # VggConvLayer(512,512,3),
            # nn.MaxPool2d(2,2,ceil_mode=True),
            # VggConvLayer(512,512,4)
        )
        
    def forward(self,x):
        return self.layers(x)
    
class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            VggConvLayer(512,256),
            nn.Upsample(scale_factor=2, mode='nearest'),
            VggConvLayer(256,256,3),
            VggConvLayer(256,128),
            nn.Upsample(scale_factor=2, mode='nearest'),
            VggConvLayer(128,128),
            VggConvLayer(128,64),
            nn.Upsample(scale_factor=2, mode='nearest'),
            VggConvLayer(64,64),
            VggConvLayer(64,3)
        )
        
    def forward(self,x):
        return self.layers(x)
    
class Adain(nn.Module):
    
    arg_v = {
        'eps':1e-5
    }
    
    def __init__(self,**kwargs) -> None:
        super().__init__()
        for key in self.arg_v.keys():
            if key in kwargs:
                setattr(self,key,kwargs[key])
            else:
                setattr(self,key,self.arg_v[key])
    
    def calculate_mean_std(self,x:torch.Tensor):
        b,c = x.size()[:2]
        x_std = (x.var(dim=-1)+self.eps).sqrt().view(b,c,1)
        x_mean = x.mean(dim=-1).view(b,c,1)
        return x_std,x_mean
    
    def forward(self,x):
        content_feat,style_feat = x
        assert content_feat.size()[:2] == style_feat.size()[:2]
        content_size = content_feat.size()
        b,c = content_size[:2]
        content_feat,style_feat = content_feat.view(b,c,-1),style_feat.view(b,c,-1)
        content_std,content_mean = self.calculate_mean_std(content_feat)
        style_std,style_mean = self.calculate_mean_std(style_feat)
        normalized_feat = (content_feat - content_mean) / content_std
        return (normalized_feat* style_std + style_mean).view(*content_size)
        
    
class AdainNetModule(pl.LightningModule):
    
    arg_v = {
        'alpha':1.0,
        'lr':1e-5,
        'train_epochs':100,
        'automatic_optimization': False,
        'content_weight': 1,
        'style_weight': 10,
    }
    
    def __init__(self, **kwargs) -> None:
        super().__init__()
        for key in self.arg_v.keys():
            if key in kwargs:
                setattr(self,key,kwargs[key])
            else:
                setattr(self,key,self.arg_v[key])
        assert self.alpha >= 0 and self.alpha <= 1
        self.encoder = VggEncoder()
        self.feature_net =  IntermediateLayerGetter(self.encoder.layers,{
            '1':'layer1_1',
            '4':'layer2_1',
            '7':'layer3_1',
            '10':'layer4_1'
        })
        self.encoder_net = IntermediateLayerGetter(self.encoder.layers,{'10':'layer4_1'})
        self.adain = Adain()
        self.decoder = Decoder()
        self.loss = nn.MSELoss()
        
    def calculate_style_loss(self,input,target):
        assert input.size() == target.size()
        b,c = input.size()[:2]
        input_std,input_mean = self.adain.calculate_mean_std(input.view(b,c,-1))
        target_std,target_mean = self.adain.calculate_mean_std(target.view(b,c,-1))
        return self.loss(input_mean,target_mean) + self.loss(input_std,target_std)
        
    def forward(self,batch, batch_index) :
        content,style = batch
        style_feats = self.feature_net(style)
        content_feats = self.encoder_net(content)
        t = self.adain([content_feats['layer4_1'],style_feats['layer4_1']])
        target = self.alpha * t + (1-self.alpha) *content_feats['layer4_1']
        g_target = self.decoder(target)
        g_target_feats = self.feature_net(g_target)
        content_loss = self.loss(g_target_feats['layer4_1'],t)
        style_loss = 0
        for key in style_feats.keys():
            style_loss += self.calculate_style_loss(style_feats[key],g_target_feats[key])
        return content_loss,style_loss
    
    def training_step(self, batch, batch_index):
        opt = self.optimizers()
        opt.zero_grad()
        content_loss,style_loss = self(batch,batch_index)
        content_loss = content_loss*self.content_weight
        style_loss = style_loss*self.style_weight
        self.log_dict({
            'loss':content_loss + style_loss,
            'content_loss': content_loss,
            'style_loss': style_loss,
        }, prog_bar=True)
        content_loss.backward(retain_graph=True)
        style_loss.backward()
        opt.step()
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.decoder.parameters(), self.lr)
        
        

