from abc import abstractmethod


class AbstractTester(object):

    @abstractmethod
    def run(self, **args):
        pass

    def __call__(self,**args):
        return self.run(**args)
