from argparse import ArgumentParser
import tester


parser = ArgumentParser()
parser.add_argument("--model", type=str, default='ImfwNetTester')
parser.add_argument("--resume_path", type=str, default=None)


args = parser.parse_args()
arg_v = vars(args)

if __name__ == '__main__':
    if arg_v['model'] in ['CameraTester','AdainCameraTester']:
        tester = getattr(tester,arg_v['model'])(arg_v['resume_path'])
    else :
        tester = getattr(tester,arg_v['model'])()
    tester.run()
