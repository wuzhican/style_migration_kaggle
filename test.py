from argparse import ArgumentParser
import tester


parser = ArgumentParser()
parser.add_argument("--model", type=str, default='ImfwNetTester')


args = parser.parse_args()
arg_v = vars(args)

if __name__ == '__main__':
    if arg_v['model'] == 'CameraTester':
        tester = getattr(tester,arg_v['model'])('./data/lightning_logs/ImfwNet/version_5/checkpoints/epoch=14-step=3885.ckpt')
    else :
        tester = getattr(tester,arg_v['model'])()
    tester.run()
