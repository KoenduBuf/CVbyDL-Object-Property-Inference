
from torch import nn
from networks import NETWORKS

NETWORKS["SEQ_128_MEDIUM"] = lambda outc: nn.Sequential(
    nn.Conv2d(3, 6, 5), nn.ReLU(),   # 3 * 128 * 128 ->  6 * 124 * 124
    nn.MaxPool2d(2, 2),              #               ->  6 *  62 *  62
    nn.Conv2d(6, 12, 7), nn.ReLU(),  #               -> 12 *  56 *  56
    nn.MaxPool2d(2, 2),              #               -> 12 *  28 *  28
    nn.Conv2d(12, 6, 5), nn.ReLU(),  #               ->  6 *  24 *  24
    nn.MaxPool2d(2, 2),              #               ->  6 *  12 *  12
    nn.Flatten(1),
    nn.Linear(6 * 12 * 12, 120), nn.ReLU(),
    nn.Linear(120, 84), nn.ReLU(),
    nn.Linear(84, outc)
)
