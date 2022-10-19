import torch,os
from torch.utils import data
from torchvision import transforms
from PIL import Image

class styleLoader(data.Dataset):
    
    def __init__(self,root_dir,transform=None,augment_ratio=10) -> None:
        super().__init__()
        self.root_dir, self.augment_ratio = root_dir, augment_ratio
        self.images=[]
        for i in os.popen('ls %s'%(root_dir)):
            i=i.replace('\n','')
            self.images.append(i)
        if transform != None:
            self.transform=transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.CenterCrop(256),
                transforms.ToTensor(),  # 转为0-1的张量
                transforms.Normalize(mean=[0.485, 0.456, 0.406],\
                    std=[0.229, 0.224, 0.225])
            ])
        
    def __getitem__(self, index) :
        idx = index % len(self.images)
        while(True):
            try:
                img = self.transform(
                    Image.open(os.path.join(self.root_dir, self.images[idx]))
                )
            except Exception as e:
                print('catch exception when load image:%s ,exception message:%s '%(self.images[idx],e))
                idx = (idx+1) % len(self.images)
                continue
            break
        return img
    
    def __len__(self):
        return self.augment_ratio*len(self.images)
        
            
        
        

