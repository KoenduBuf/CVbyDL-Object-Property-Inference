#!/usr/bin/env python

import os
from torch.utils.data import Dataset
from PIL import Image

def img_file_extension(filename):
    ext = os.path.splitext(filename)[1]
    if not isinstance(ext, str):
        ext = ext.decode('utf-8')
    return ext.lower() in { '.jpg', '.jpeg', '.png' }


class FruitImage:
    def __init__(self, from_file, on_types):
        base_name    = os.path.basename(from_file)
        name_parts   = base_name.split('_')
        self.typei   = on_types.index(name_parts[0])\
            if name_parts[0] in on_types else -1
        self.weight  = int(name_parts[1][:-1])
        self.index   = int(name_parts[2])
        self.goodness= int(name_parts[3][0])
        self.file    = from_file
        self.image   = None


class FruitImageDataset(Dataset):
    DEFAULT_TYPES = ("apple", "banana", "kiwi",
        "onion", "tomato", "orange", "mandarin")

    def __init__(self, folder, types=DEFAULT_TYPES, transform=None):
        self.types = types
        self.fruit_images = [ ]
        self.transform = transform
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
        if self.transform:
            imgdata = self.transform(imgdata)
        return ( imgdata, fi.typei )

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
        nw = FruitImageDataset(None, self.types, self.transform)
        new_self = [ ]
        counters = [ 0 ] * len(self.types)
        for fi in self.fruit_images:
            counters[fi.typei] += 1
            if counters[fi.typei] % n == 0:
                nw.fruit_images.append(fi)
            else: new_self.append(fi)
        self.fruit_images = new_self
        return nw


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
