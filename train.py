from __future__ import absolute_import, division, print_function

from options import FMCSOptions
from engine import TrainEngine

options = FMCSOptions()
opts = options.parse()


if __name__ == "__main__":
    trainer = TrainEngine(opts)
    trainer.train()