#!/usr/bin/env python3

import os
from PIL import Image


class Resizer:
    def __init__(self, folder, tofolder):
        self.folder = folder
        self.tofolder = tofolder

    def todo_images(self):
        foldere = os.fsencode(self.folder)
        for file in os.listdir(foldere):
            filename = os.fsdecode(file)
            fromfile = os.path.join(self.folder, filename)
            if os.path.isdir(fromfile): continue
            yield (fromfile, filename)

    def sizes_summary(self):
        minw = minh = float('inf')
        maxw = maxh = float('-inf')
        for img_from, _ in self.todo_images():
            img = Image.open(img_from)
            minw = min(img.width, minw)
            minh = min(img.height, minh)
            maxw = max(img.width, maxw)
            maxh = max(img.height, maxh)
        return (minw, minh, maxw, maxh)

    def autoresize(self, tosize):
        tof = os.path.join(self.tofolder,
            f"auto{tosize[0]}x{tosize[1]}")
        os.makedirs(tof, exist_ok=True)
        for img_from, filename in self.todo_images():
            img = Image.open(img_from)
            resized = img.resize(tosize)
            tofile = os.path.join(tof, filename)
            resized.save(tofile)
        return tof


if __name__ == '__main__':
    resizer = Resizer('../images', '../images')
    minw, minh, maxw, maxh = resizer.sizes_summary()
    print(f"Width: {minw}-{maxw} | Height: {minh}-{maxh}")
    resizer.autoresize( (64, 64) )
