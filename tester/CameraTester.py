import models,utils,cv2,numpy,time,torch,threading
from tester.AbstractTester import AbstractTester
from torchvision import transforms

class VideoCapturer(threading.Thread):
    def __init__(self,video):
        threading.Thread.__init__(self)
        self.cap = cv2.VideoCapture(video)
        self.read_lock = threading.Lock()
        
    def read_frame(self):
        self.read_lock.acquire()
        res,frame = self.res,self.frame
        self.read_lock.release()
        return res,frame 
    
    def run(self):
        while self.cap.isOpened():
            self.read_lock.acquire()
            self.res,self.frame = self.cap.read()
            self.read_lock.release()
            time.sleep(1/20)
        self.cap.release()

class CameraTester(AbstractTester):
    args_v = {
        'video':0,
        'video_tag':'video',
        'transform': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'to_numpy': transforms.Compose([
            transforms.Lambda(lambda x:utils.UnNormalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))(x)),
            transforms.Lambda(lambda x:x.clamp(0,1)),
            transforms.ToPILImage(),
            transforms.Lambda(lambda x:cv2.cvtColor(numpy.asarray(x),cv2.COLOR_RGB2BGR))
        ])
    }
    
    def __init__(self,checkpoint_path,**args) -> None:
        super().__init__()
        for key in self.args_v:
            if key in args.keys():
                setattr(self,key,args[key])
            else:
                setattr(self,key,self.args_v[key])
        self.fwnet = models.FWNetModule.load_from_checkpoint(
            checkpoint_path,
            style = utils.load_image('./data/style.jpeg')
        )
        self.cap = VideoCapturer(self.video)
        self.cap.start()
        self.fwnet.cuda()
        self.fwnet.eval()
        
    def run(self):
        while True:
            a = time.time()
            ret,frame = self.cap.read_frame()
            if(ret):
                torch.cuda.empty_cache()
                frame = cv2.flip(frame,1)
                img = self.transform(frame).cuda()
                target = self.fwnet.fwNet(img).cpu()
                show_frame = self.to_numpy(target)
                seconds = time.time() - a
                cv2.putText(show_frame,"fps:%s"%(1/seconds),(5,50),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                cv2.imshow(self.video_tag,show_frame)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                continue
            else:
                break
        self.cap.cap.release()
        cv2.destroyAllWindows()
            
                
                
                
            
        
