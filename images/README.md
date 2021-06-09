
# Our dataset

Format of files: LABEL_WEIGHTg_NR_GOODNESS.jpg

 - LABEL:    A label for which fruit it is
 - WEIGHT:   A label for how heavy this piece was
 - NR:       Just a number to prevent duplicate names
 - GOODNESS: The goodness of the image, i.e. how clear it is,
 this will be a number of 1-4 meaning:
  - 1: Fruit at least 1/16th of the image, clear in view
  - 2: Fruit at least 1/36th of the image, mostly clear
  - 3: Fruit at smaller than 1/36th of the image, or unclear
  - 4: Unused, because of reasons like: more than 1 fruit
