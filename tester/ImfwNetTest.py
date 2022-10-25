import models
import utils
from tester.AbstractTester import AbstractTester
import matplotlib.pyplot as plt
from torchvision import transforms

from utils.Functions import show_tensor


class ImfwNetTester(AbstractTester):
    def __init__(self) -> None:
        self.checkpoint_path = utils.GetResumePath('./data/lightning_logs/ImfwNet')
        self.fwnet = models.FWNetModule.load_from_checkpoint(
            self.checkpoint_path, style_wuzhican=utils.load_image('./data/style.jpeg'))
        self.un = utils.UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.toImg = transforms.ToPILImage()

    def run(self):
        un = utils.UnNormalize((0.229, 0.224, 0.225),(0.485, 0.456, 0.406))
        toimg = transforms.ToPILImage()
        img = utils.load_image('./data/validation.jpg')
        # show_tensor(img,lambda x:toimg(un(x)).show())
        show_tensor(img)