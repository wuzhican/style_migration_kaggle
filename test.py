from argparse import ArgumentParser
import tester


parser = ArgumentParser()
parser.add_argument("--model", type=str, default='ImfwNetTester')


args = parser.parse_args()
arg_v = vars(args)

if __name__ == '__main__':
    if arg_v['model'] == 'CameraTester':
        tester = getattr(tester,arg_v['model'])('./data/lightning_logs/ImfwNet/version_5/checkpoints/epoch=14-step=3885.ckpt')
    elif arg_v['model'] == 'AdainCameraTester':
        tester = getattr(tester,arg_v['model'])('./data/lightning_logs/AdainNet/version_0/checkpoints/epoch=2-step=1557.ckpt')
    else :
        tester = getattr(tester,arg_v['model'])()
    tester.run()
