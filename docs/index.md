<link type="text/css" rel="stylesheet" href="style.css" />

Computer vision nowadays is dominated by deep learning using Convolutional Neural Networks (CNNs), for this project we try to learn about the usage of CNNs to solve a problem in various ways

# Goals, and our dataset

For the goal of this project we try to create a CNN that can estimate the weight of various classes of fruit, and to try different approaches to find the machine learning model with the highest accuracy for this task.

Our dataset for this problem consists of homemade pictures of fruits. The dataset contains 7 fruit classes, `apple`, `banana`, `kiwi`, `onion`, `tomato`, `orange`, and `mandarin`, for each of these classes we have around 190 square pictures from various angles including different sceneries and images where the fruit is partially obscured. All of the images are labeled with the type of fruit in the picture and the weight of that piece of fruit, a few example of the banana class and a quick table on our dataset are given below.

<span class="note">Note: For the squaring of these images, we cropped them in such a way that as little of the fruit as possible was cut from the images, for this we used a little program we wrote that is publicly available [here](https://github.com/KoenduBuf/tk-imgdecide).</span>

| A banana of 148 grams | A banana of 173 grams | A banana of 188 grams |
| :-------------------: | :-------------------: | :-------------------: |
| ![banana 148g](https://koendubuf.github.io/CVbyDL-Object-Property-Inference/images/banana_148g_5_1.jpg) |  ![banana 173g](https://koendubuf.github.io/CVbyDL-Object-Property-Inference/images/banana_173g_6_1.JPG) | ![banana 173g](https://koendubuf.github.io/CVbyDL-Object-Property-Inference/images/banana_188g_3_1.jpg) |

<br />

| class    | #Images | #Unique weights |  Weight: min-avg-max |
|:--------:|:-------:|:---------------:|:--------------------:|
| apple    |    195  |    13           |  165 - 188.7 - 206   |
| banana   |    190  |    18           |  139 - 170.2 - 193   |
| kiwi     |    101  |     5           |   97 -  99.0 - 103   |
| onion    |    142  |    14           |   68 - 128.3 - 179   |
| tomato   |    191  |    12           |   71 - 130.3 - 177   |
| orange   |    196  |    16           |  151 - 174.7 - 209   |
| mandarin |     91  |     6           |   98 - 106.2 - 118   |

# Training and test sets

If we would train our ML models on the full dataset, it would be unfair to use the same images...

# A first test: Fruit classification + weight averages

We made a CNN that classified the images... results...

So we can classify fruits, that means that we already have the most simple CNN for weight estimation, we can have our fruit classifier guess the fruit and then take the average weight of that fruit as our weight estimation. Using this naive approach we get results...

# Attempt two: Weight range classifier

Next we tried to train a single CNN to classify the fruits again, but this time we tried to classify them by their weight range. This means that instead of 1 label for each fruit class, we created classes for every N grams (0-N, N-2N, 2N-3N, etc...), results...

# Using multiple networks?

...
