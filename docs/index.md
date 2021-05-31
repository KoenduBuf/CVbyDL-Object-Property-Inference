<link type="text/css" rel="stylesheet" href="style.css">
<link rel="shortcut icon" type="image/png" href="images/favicon.png">


Computer vision nowadays is dominated by deep learning using Convolutional Neural Networks (CNNs), for this project we learn about the usage of CNNs to solve a problem in various ways, from basic classifiers to using multiple networks for a single instance.

# Goals, and our dataset

For the goal of this project we try to create a CNN that can estimate the weight of various classes of fruit, and to try different approaches to find the machine learning model with the highest accuracy for this task.

Our dataset for this problem consists of homemade pictures of fruits. The dataset contains 7 fruit classes, `apple`, `banana`, `kiwi`, `onion`, `tomato`, `orange`, and `mandarin`, for each of these classes we have around 190 square pictures from various angles including different sceneries and images where the fruit is partially obscured. All of the images are labeled with the type of fruit in the picture and the weight of that piece of fruit, a few example of the banana class and a quick table on our dataset are given below.

<span class="note"><b>Quick detail: </b>Originally we forgot to take square images, we cropped them after taking pictures in such a way that as little of the fruit as possible was cut from the images. Normally this would be time consuming, so we created a little program that is publicly available [here](https://github.com/KoenduBuf/tk-imgdecide).</span>

| A banana of 148 grams | A banana of 173 grams | A banana of 188 grams |
| :-------------------: | :-------------------: | :-------------------: |
| ![banana 148g](https://koendubuf.github.io/CVbyDL-Object-Property-Inference/images/banana_148g_5_1.jpg) |  ![banana 173g](https://koendubuf.github.io/CVbyDL-Object-Property-Inference/images/banana_173g_6_1.JPG) | ![banana 173g](https://koendubuf.github.io/CVbyDL-Object-Property-Inference/images/banana_188g_3_1.jpg) |

<p></p>

| class    | #Images | #Unique weights |  Weight: min-avg-max |
|:--------:|:-------:|:---------------:|:--------------------:|
| apple    |    195  |    13           |  165 - 188.7 - 206   |
| banana   |    190  |    18           |  139 - 170.2 - 193   |
| kiwi     |    181  |    12           |   84 -  95.1 - 103   |
| onion    |    192  |    17           |   68 - 137.7 - 193   |
| tomato   |    191  |    12           |   71 - 130.3 - 177   |
| orange   |    196  |    16           |  151 - 174.7 - 209   |
| mandarin |    171  |    12           |   96 - 106.2 - 122   |

# Training and test sets

Training our ML models on the whole dataset would be unfair since we can then not test its performance on unseen data. For this reason we divided our dataset into a train set and a test set, where the train set contains 90% of the dataset. The models are only allowed to train on the train set, and evaluated on the unseen test set.

With the already small dataset, and the split into a train and test set, we have very little data left in the train set for the network to learn from. Our solution to this problem was data augmentation. Specifically any training instance is first turned into 4 separate instances by applying a vertical flip of the image, and then a horizontal flip, both with 50% chance. This strategy already gives a significant amount more unique training instances. Finally we randomly remove a section of the image again with a 50% chance, this gives us more than enough instances to train on, this approach was based on [this paper](https://arxiv.org/abs/1708.04896).

# A first test: Fruit classification + weight averages

We made a CNN that classified the images just by their fruit class. For this classifier we tried various network architectures, trained from scratch. The architecture that got the best performance reached a <span class="tooltip"> classification accuracy of 70% <span class="tooltiptext">In perspective: random guessing would give a 1/7 = 14% classification accuracy</span> </span>, which consisted of N layers ....

So we can classify fruits, that means that we already have the most simple CNN for weight estimation, we can have our fruit classifier guess the fruit and then take the average weight of that fruit as our weight estimation. Using this naive approach we get results...

# Attempt two: Weight range classifier

Next we tried to train a single CNN to classify the fruits again, but this time we tried to classify them by their weight range. This means that instead of 1 label for each fruit class, we created classes for every N grams (0-N, N-2N, 2N-3N, etc...), results...

# Using multiple networks?

Things...
