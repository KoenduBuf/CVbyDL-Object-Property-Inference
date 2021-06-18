
NETWORKS = { }

import torch
from networks.BaseNets import *
from networks.TransferNet import *

def get_network(name, outc, on_device="cpu", flatten=False):
    print(f"Getting network {name}, to run on {on_device}")
    net = NETWORKS[name](outc)
    if flatten: # make (n,1) into (n) shape
        net = torch.nn.Sequential(net, torch.nn.Flatten(0))
    net.to(on_device)
    return net
