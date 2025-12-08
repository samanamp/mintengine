from math import sqrt
from pathlib import Path
import torch
import torch.nn as nn

from mintengine.models.gemma3 import Gemma3


def main():
    gemma_model = Gemma3()
    out = gemma_model.generate("Tell me a story")
    print(out)


if __name__ == "__main__":
    main()
