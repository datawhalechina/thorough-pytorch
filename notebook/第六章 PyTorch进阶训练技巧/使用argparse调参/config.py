import argparse
import json
# todo: 尝试下json换成yaml，pickle等库


class Options():
    """Class to manage options using parser and namespace.
    """
    def __init__(self):
        self.initialized = False

    def add_arguments_parser(self, parser: argparse.ArgumentParser):
        """Add a set of arguments to the parser.
        Parameters
        ----------
        parser : argparse.ArgumentParser
            Parser to add arguments to.
        Returns
        -------
        parser : argparse.ArgumentParser
            Parser with added arguments.
        """
        parser.add_argument('--batch_size', type=int, default=32, help="input batch size,default = 64")
        parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for, default=10')
        parser.add_argument("--seed", type=int, default=66, help="random seed")
        parser.add_argument("--class_num", type=int, default=30, help="classification category,default 10")
        parser.add_argument("--log_path", type=str, default="./runs/log")
        parser.add_argument("--img_size", type=tuple, default=(224, 224), help="input image size")
        parser.add_argument("--show", action="store_true", default=False)
        self.initialized = True

        return parser

    def _initialize_options(self):
        """Initialize a namespace that store options.
        Returns
        -------
        opt: argparse.Namespace
            Namespace with options.
        """
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.add_arguments_parser(parser)
            self.parser = parser

        else:
            print("WARNING: Options was already initialized before")

        return self.parser.parse_args()

    def parse(self):
        """Initialize a namespace that store options.
        Returns
        -------
        opt: argparse.Namespace
            Namespace with options.
        """
        opt = self._initialize_options()
        return opt

    def print_options(self, opt: argparse.Namespace):
        """Print all options and the default values (if changed).
        Parameters
        ----------
        opt : argparse.Namespace
            Namespace with options to print.
        """
        # create a new parser with default arguments
        parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.add_arguments_parser(parser)

        message = '----------------- Options ---------------\n'
        for key, value in sorted(vars(opt).items()):
            comment = ''
            default = parser.get_default(key)
            if value != default:
                comment = f'(default {default})'
            key, value = str(key), str(value)
            message += f'{key}: {value} {comment}\n'
        print(message)

    def save_options(self, opt: argparse.Namespace, path: str):
        """Save options to a json file.
        Parameters
        ----------
        opt : argparse.Namespace
            Namespace with options to save.
        path : str
            Path to save the options (.json extension will
            be automatically added at the end if absent).
        """
        if not path.endswith('.json'):
            path += '.json'
        with open(path, 'w') as f:
            f.write(json.dumps(vars(opt), indent=4))

    def load_options(self, path:str):
        # bug:这玩意当你的required = True，你必须输入必需参数，不过load的话可以直接load进网络，不要load进入argparse？
        """Load options from a json file.
        Parameters
        ----------
        path : str
            Path to load the options (.json extension will
            be automatically added at the end if absent).
        Returns
        -------
        opt : argparse.Namespace
            Namespace with loaded options.
        """
        if not path.endswith('.json'):
            path += '.json'
        # init a new namespace with default arguments
        parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        opt = self.add_arguments_parser(parser).parse_args([])

        variables = json.load(open(path, 'r'))
        for key, value in variables.items():
            setattr(opt, key, value)
        print("--------------load options finish----------------")
        return opt


if __name__ == '__main__':
    options = Options()
    opt = options.load_options("./options.json")
    if not opt.show:
        options.print_options(opt)
