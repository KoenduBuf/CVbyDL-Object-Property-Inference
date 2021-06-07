#!/usr/bin/env python

import os
import torchvision
import numpy as np
from torch.utils import data
from PIL import Image


TRANSFORMS_TEST  = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

TRANSFORMS_TRAIN = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomVerticalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.RandomErasing(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def img_file_extension(filename):
    ext = os.path.splitext(filename)[1]
    if not isinstance(ext, str):
        ext = ext.decode('utf-8')
    return ext.lower() in { '.jpg', '.jpeg', '.png' }


class FruitImage:
    property_transforms = {
        "class":      lambda: lambda fi: fi.typei,
        "weight":     lambda: lambda fi: (fi.weight - 50) / 250, # weight, 0-1
        "weight_win": lambda size: lambda fi: int(fi.weight / int(size)),
        "weight_grp": lambda amnt: lambda fi: int(fi.weight /(250/int(amnt)))
    }

    def __init__(self, from_file, on_types):
        base_name    = os.path.basename(from_file)
        name_parts   = base_name.split('_')
        self.typei   = on_types.index(name_parts[0])\
            if name_parts[0] in on_types else -1
        self.weight  = int(name_parts[1][:-1])
        self.goodness= int(name_parts[3][0])\
            if len(name_parts) >= 4 else 0
        self.file    = from_file
        self.image   = None


class FruitImageDataset(data.Dataset):
    DEFAULT_TYPES = ("apple", "banana", "kiwi",
        "onion", "tomato", "orange", "mandarin")

    def __init__(self, folder, types=DEFAULT_TYPES, img_transform=None,
        lbl_transform=lambda fi: fi.typei):
        self.types = types
        self.fruit_images = [ ]
        self.img_transform = img_transform
        self.lbl_transform = lbl_transform
        if folder is None: return
        foldere = os.fsencode(folder)
        for file in os.listdir(foldere):
            filename = os.fsdecode(file)
            if not img_file_extension(filename): continue
            path = os.path.join(folder, filename)
            if not os.path.isfile(path): continue
            fi = FruitImage(path, self.types)
            if fi.typei == -1: continue     # Dont use other types of fruit
            if fi.goodness > 2: continue    # Dont use bad quality images
            self.fruit_images.append(fi)

    def __len__(self):
        return len(self.fruit_images)

    def __getitem__(self, index):
        fi = self.fruit_images[index]
        if fi.image is not None:
            return ( fi.image, fi.typei )
        imgdata = Image.open(fi.file)
        if self.img_transform:
            imgdata = self.img_transform(imgdata)
        lbldata = fi
        if self.lbl_transform:
            lbldata = self.lbl_transform(lbldata)
        return ( imgdata, lbldata )

    def summary_of_typei(self, ti):
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

    def split_1_in_n(self, n=10):
        nw = FruitImageDataset(None, self.types,
            self.img_transform, self.lbl_transform)
        new_self = [ ]
        counters = [ 0 ] * len(self.types)
        for fi in self.fruit_images:
            counters[fi.typei] += 1
            if counters[fi.typei] % n == 0:
                nw.fruit_images.append(fi)
            else: new_self.append(fi)
        self.fruit_images = new_self
        return nw

    def show_sample(self, amount=6):
        # show random images and print labels
        import matplotlib.pyplot as plt
        showloader = data.DataLoader(self, shuffle=True,
            batch_size=amount, num_workers=2)
        images, labels = iter(showloader).next()
        classes = FruitImageDataset.DEFAULT_TYPES
        print(' '.join(classes[lbl] for lbl in labels))
        grid_img = torchvision.utils.make_grid(images)
        grid_img = grid_img / 2 + 0.5 # unnormalize
        plt.imshow(np.transpose(grid_img.numpy(), (1, 2, 0)))
        plt.show()


def get_datasets(lbl_transform="class", print_tables=True):
    lbl_transform, *lbl_trans_args = lbl_transform.split()
    if lbl_transform in FruitImage.property_transforms:
        lbl_transform = FruitImage.property_transforms[lbl_transform]
        lbl_transform = lbl_transform(*lbl_trans_args)
    train_set = FruitImageDataset("../images/auto128x128",
        img_transform=TRANSFORMS_TRAIN, lbl_transform=lbl_transform)
    test_set  = train_set.split_1_in_n(10)
    test_set.img_transform = TRANSFORMS_TEST
    if print_tables:
        print_summary_tables(
            (train_set, "TRAIN"),
            (test_set, "TEST") )
    return train_set, test_set


def dataset_summary_table(dataset, name):
    returnlist = [ name.ljust(11) +
        f"Amount/Uniq     Weight min-avg-max", " -" * 22 + " " ]
    for i, t in enumerate(FruitImageDataset.DEFAULT_TYPES):
        s = dataset.summary_of_typei(i)
        returnlist.append(  t.ljust(11) + ( str(s['amount']).rjust(5)
        + " / " + str(s['unique']).rjust(2) ) + ( str(s['min']).rjust(3)
        + " - " + str(s['avg_weight']).rjust(6) + " - " + str(s['max'])
        .rjust(3) ).rjust(24) )
    return returnlist


def print_summary_tables(*set_name_tuples, side_to_side=2):
    side_to_side_builder = None
    for counter, (set, name) in enumerate(set_name_tuples):
        # Get the text, and paste it next to whats there
        summ_text = dataset_summary_table(set, name)
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


if __name__=="__main__":
    ds = FruitImageDataset("../images")
    print_summary_tables( (ds, "") )
