
import os
import torch
import torchvision
import numpy as np
from utils.ImageResizer import *
from torch.utils import data
from PIL import Image
import random


TRANSFORMS_BASE = lambda norm_mean, norm_std: torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(norm_mean, norm_std)
])

TRANSFORMS_AUG  = torch.nn.Sequential(
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomVerticalFlip(),
    torchvision.transforms.RandomErasing()
)

WEIGHT_MIN, WEIGHT_MAX = 68, 209
WEIGHT_RANGE = WEIGHT_MAX - WEIGHT_MIN


class FruitImage:
    def __init__(self, from_file, on_types):
        base_name    = os.path.basename(from_file)
        name_parts   = base_name.split('_')
        self.typei   = on_types.index(name_parts[0])\
            if name_parts[0] in on_types else -1
        self.weight  = int(name_parts[1][:-1])
        self.goodness= int(name_parts[3][0])\
            if len(name_parts) >= 4 else 0
        self.file    = from_file
        # Caching things for da speed
        self.image   = None
        self.lbl     = None


class FruitImageDataset(data.Dataset):
    DEFAULT_TYPES = ("apple", "banana", "kiwi",
        "onion", "tomato", "orange", "mandarin")

    def __init__(self, folder, img_transform_base=None,
        img_transform_aug=None, img_to_device="cpu",
        lbl_transform=lambda fi: fi.weight, types=DEFAULT_TYPES):
        self.types              = types
        self.fruit_images       = [ ]
        self.img_to_device      = img_to_device
        self.img_transform_aug  = img_transform_aug
        # Check which files to include in this set
        if folder is None: return
        foldere = os.fsencode(folder)
        for file in os.listdir(foldere):
            filename = os.fsdecode(file)
            # Check if the file is an image
            if not img_file_extension(filename): continue
            path = os.path.join(folder, filename)
            if not os.path.isfile(path): continue
            fi = FruitImage(path, self.types)
            # Check if we want this one
            if fi.typei == -1: continue     # Dont use other types of fruit
            if fi.goodness > 2: continue    # Dont use bad quality images
            self.fruit_images.append(fi)
        # Load all images, normalize and maybe on GPU even
        random.shuffle(self.fruit_images)
        self.lbl_transform = lbl_transform # do these last, to compute
        self.img_transform_base = img_transform_base
        self.to_device()

    @property
    def lbl_transform(self):
        return self._lbl_transform

    @lbl_transform.setter
    def lbl_transform(self, value):
        self._lbl_transform = value
        for fi in self.fruit_images:
            fi.lbl = torch.tensor(self._lbl_transform(fi))

    @property
    def img_transform_base(self):
        return self._img_transform_base

    @img_transform_base.setter
    def img_transform_base(self, value):
        self._img_transform_base = value
        for fi in self.fruit_images:
            imgdata = Image.open(fi.file)
            fi.image = imgdata if self._img_transform_base\
            is None else self.img_transform_base(imgdata)

    def to_device(self, device=None):
        if device is None: device = self.img_to_device
        for fi in self.fruit_images:
            fi.image = fi.image.to(device)
            fi.lbl   = fi.lbl.to(device)

    def __len__(self):
        return len(self.fruit_images)

    def __getitem__(self, index):
        fi = self.fruit_images[index]
        imgdata = fi.image
        if self.img_transform_aug:
            imgdata = self.img_transform_aug(imgdata)
        return ( imgdata, fi.lbl )

    def split_1_in_n(self, n=10, seed=0):
        nw = FruitImageDataset(None, self.img_transform_base,
            self.img_transform_aug, self.img_to_device,
            self.lbl_transform, self.types)
        new_self = [ ]
        counters = [ seed ] * len(self.types)
        for fi in self.fruit_images:
            counters[fi.typei] += 1
            if counters[fi.typei] % n == 0:
                nw.fruit_images.append(fi)
            else: new_self.append(fi)
        self.fruit_images = new_self
        return nw

    def summary_of_typei(self, ti):
        if isinstance(ti, str): ti = self.types.index(ti)
        of_fruit = filter(lambda fi: fi.typei == ti, self.fruit_images)
        their_weights = list(map(lambda fi: fi.weight, of_fruit))
        total_weight = sum(their_weights)
        amount = len(their_weights)
        unique = len( set(their_weights) )
        return { 'amount': amount, 'unique': unique,
            'avg_weight': round(total_weight/amount, 2)
                if amount > 0 else 0,
            'min': min(their_weights, default=0),
            'max': max(their_weights, default=0) }

    def dataset_summary_table(self, name):
        returnlist = [ name.ljust(11) +
            f"Amount/Uniq     Weight min-avg-max", " -" * 22 + " " ]
        for i, t in enumerate(self.types):
            s = self.summary_of_typei(i)
            returnlist.append(  t.ljust(11) + ( str(s['amount']).rjust(5)
            + " / " + str(s['unique']).rjust(2) ) + ( str(s['min']).rjust(3)
            + " - " + str(s['avg_weight']).rjust(6) + " - " + str(s['max'])
            .rjust(3) ).rjust(24) )
        return returnlist

    def show_sample(self, amount=6):
        # show random images and print labels
        import matplotlib.pyplot as plt
        showloader = data.DataLoader(self, shuffle=True,
            batch_size=amount, num_workers=2)
        images, labels = iter(showloader).next()
        classes = self.types
        print(' '.join(classes[lbl] for lbl in labels))
        grid_img = torchvision.utils.make_grid(images)
        grid_img = grid_img / 2 + 0.5 # unnormalize
        plt.imshow(np.transpose(grid_img.numpy(), (1, 2, 0)))
        plt.show()


# Get a test and a dataset, resize images if needed. Norm from ResNet
def get_datasets(lbl_transform, image_wh=128, device="cpu", print_tables=True,
    norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225)):
    # If the dataset is not there yet, then make it
    resizer = Resizer('../images', '../images')
    to_folder = resizer.autoresize(image_wh)
    # Then read in the images and split them up, train has aug, test doesn't
    base_transform = TRANSFORMS_BASE(norm_mean, norm_std)
    train_set = FruitImageDataset(to_folder, base_transform,
        TRANSFORMS_AUG, device, lbl_transform)
    test_set  = train_set.split_1_in_n(10)
    test_set.img_transform_aug = None
    if print_tables:
        print_summary_tables(
            (train_set, "TRAIN"),
            (test_set, "TEST") )
    return train_set, test_set


################################################################################
######################################################### Printing pretty tables

def print_summary_tables(*set_name_tuples, side_to_side=2):
    side_to_side_builder = None
    for counter, (set, name) in enumerate(set_name_tuples):
        # Get the text, and paste it next to whats there
        summ_text = set.dataset_summary_table(name)
        if side_to_side_builder is None:
            side_to_side_builder = summ_text
        else:
            side_to_side_builder = list(map(
            lambda tpl: tpl[0] + "    |    " + tpl[1],
            zip(side_to_side_builder, summ_text)))
        # Print it whenever we have enough text
        if counter + 1 % side_to_side == 0:
            print("\n".join(side_to_side_builder) + "\n")
            side_to_side_builder = None
    # Print last one
    if side_to_side_builder is not None:
        print("\n".join(side_to_side_builder))
