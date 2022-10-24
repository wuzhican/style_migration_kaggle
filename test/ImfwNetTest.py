import models,utils
from test.AbstractTester import AbstractTester
import matplotlib.pyplot as plt
from torchvision import transforms

class ImfwNetTester(AbstractTester):
    def __init__(self) -> None:
        self.checkpoint_path = utils.GetResumePath('./data/lightning_logs/ImfwNet')
        self.fwnet = models.FWNetModule.load_from_checkpoint(
            self.checkpoint_path, style_wuzhican=utils.load_image('./data/style.jpeg'))
        self.un = utils.UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.toImg = transforms.ToPILImage()
        
    def run(self):
        img = utils.load_image('./data/style1.jpg')[0]
        target = self.toImg(self.un(self.fwnet.fwNet(img).clamp(0,1)))
        target.show()