#!/usr/bin/env python

import os
from torch.utils.data import Dataset
from PIL import Image

class FruitImage:
    def __init__(self, from_file, on_types):
        base_name    = os.path.basename(from_file)
        name_parts   = base_name.split('_')
        self.typei   = on_types.index(name_parts[0])\
            if name_parts[0] in on_types else -1
        self.weight  = int(name_parts[2][:-1])
        self.index   = name_parts[3]
        self.file    = from_file
        self.image   = None


class FruitImageDataset(Dataset):
    DEFAULT_TYPES = ("apple", "banana", "kiwi",
        "union", "tomato", "orange", "mandarin")

    def __init__(self, folder, types=DEFAULT_TYPES, transform=None):
        self.types = types
        self.fruit_images = [ ]
        self.transform = transform
        if folder is None: return
        foldere = os.fsencode(folder)
        for file in os.listdir(foldere):
            filename = os.fsdecode(file)
            path = os.path.join(folder, filename)
            if not os.path.isfile(path): continue
            fi = FruitImage(path, self.types)
            if fi.typei == -1: continue
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
        return { 'amount': amount, 'unique': unique, 'avg_weight':
            total_weight / amount if amount > 0 else 0,
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



def dataset_summary_table(dataset):
    print(f"Type        Amount      Unique      Weight range"
        + "      Average Weight\n" + (" -" * 34) )
    for i, t in enumerate(FruitImageDataset.DEFAULT_TYPES):
        s = ds.summary_of_typei(i)
        print(t.ljust(12, ' ')                  # Type name
        + str(s['amount']).ljust(12, ' ')       # Amount
        + str(s['unique']).ljust(12, ' ')       # Unique
        + ( str(s['min']) + "-" + str(s['max']) ).ljust(18, ' ')
        + str(round(s['avg_weight'], 2)).ljust(10, ' '))

if __name__=="__main__":
    ds = FruitImageDataset("../images")
    dataset_summary_table(ds)
