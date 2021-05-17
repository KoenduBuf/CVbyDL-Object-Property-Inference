#!/usr/bin/env python

import os
from torch.utils.data import Dataset


class FruitImage:
    def __init__(self, from_file):
        base_name   = os.path.basename(from_file)
        name_parts  = base_name.split('_')
        self.type   = name_parts[0]
        self.weight = int(name_parts[1].split('g')[0])
        self.index  = name_parts[1].split(' ')[1]
        self.file   = from_file


class FruitImageDataset(Dataset):
    DEFAULT_TYPES = ("apple", "banana", "kiwi",
        "union", "tomato", "orange", "manderin")

    def __init__(self, folder, types=DEFAULT_TYPES):
        self.types = types
        self.fruit_images = [ ]
        foldere = os.fsencode(folder)
        for file in os.listdir(foldere):
            filename = os.fsdecode(file)
            path = os.path.join(folder, filename)
            fi = FruitImage(path)
            if not fi.type in types: continue
            self.fruit_images.append(fi)

    def __len__(self):
        return len(self.fruit_images)

    def __getitem__(self, index):
        fi = self.fruit_images[index]
        return fi

    def summary_of_type(self, t):
        of_fruit = filter(lambda fi: fi.type == t, self.fruit_images)
        their_weights = list(map(lambda fi: fi.weight, of_fruit))
        total_weight = sum(their_weights)
        amount = len(their_weights)
        unique = len( set(their_weights) )
        return { 'amount': amount, 'unique': unique, 'avg_weight':
            total_weight / amount if amount > 0 else 0,
            'min': min(their_weights, default=0),
            'max': max(their_weights, default=0) }


def dataset_summary_table(dataset):
    print(f"Type        Amount      Unique      Weight range"
        + "      Average Weight\n" + (" -" * 34) )
    for t in FruitImageDataset.DEFAULT_TYPES:
        s = ds.summary_of_type(t)
        print(t.ljust(12, ' ')                  # Type name
        + str(s['amount']).ljust(12, ' ')       # Amount
        + str(s['unique']).ljust(12, ' ')       # Unique
        + ( str(s['min']) + "-" + str(s['max']) ).ljust(18, ' ')
        + str(round(s['avg_weight'], 2)).ljust(10, ' '))


if __name__=="__main__":
    ds = FruitImageDataset("../images")
    dataset_summary_table(ds)



