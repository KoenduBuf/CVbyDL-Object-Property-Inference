
NETWORKS = { }

from torch import nn
from networks.BaseNets import *
from networks.TransferNet import *

def get_network(name, outc, flatten=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Getting network {name}, running on {device}")
    net = NETWORKS[name](outc)
    if flatten: # make (n,1) into (n) shape
        net = nn.Sequential(net, nn.Flatten(0))
    net.to(device)
    return net
