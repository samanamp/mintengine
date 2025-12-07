from math import sqrt
from pathlib import Path
import torch
import torch.nn as nn

from mintengine.models.gemma3 import Gemma3


def main():
    gemma_model = Gemma3()
    gemma_model.generate("Hello World!")


if __name__ == "__main__":
    main()
