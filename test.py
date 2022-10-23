from argparse import ArgumentParser
import test


parser = ArgumentParser()
parser.add_argument("--model", type=str, default='ImfwNetTester')


args = parser.parse_args()
arg_v = vars(args)

if __name__ == '__main__':
    tester = getattr(test,arg_v['model'])()
    tester.run()
