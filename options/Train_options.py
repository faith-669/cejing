from argparse import ArgumentParser


class TrainOptions:
    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument('', type=, default=, help=)
        self.parser.add_argument('', type=, default=, help=)




    def parse(self):
        opts = self.parser.parse_args()
        return opts