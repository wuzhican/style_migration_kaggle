from test.AbstractTester import AbstractTester
import models,utils
import matplotlib.pyplot as plt
from torchvision import transforms
from test.AbstractTester import AbstractTester

class MSNetTester(AbstractTester):
    def __init__(self) -> None:
        super().__init__()
        # self.checkpoint_path = utils.GetResumePath('./data/lightning_logs/SMNet')
        self.checkpoint_path = './checkpoint/lightning_logs/epoch=44-step=5805.ckpt'
        self.smnet = models.SMNet.load_from_checkpoint(
            self.checkpoint_path, style=utils.load_image('./data/style2.jpg',shape=(512,512)))
    
    def run(self, **args):
        show_image = self.smnet.input_image.clamp(0,1)
        toImg = transforms.ToPILImage()
        un = utils.UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        target_image = toImg(show_image[0])
        target_image.show()
