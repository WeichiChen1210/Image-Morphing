# Image-Morphing
Implementation of image morphing based on the paper: [Feature-Based Image Metamorphosis](https://www.cs.princeton.edu/courses/archive/fall00/cs426/papers/beier92.pdf)

This work morphs the given two images and produce a short animation from src img to dst img.

## Requirement
This work is done using python 3.6.
+ numpy==1.18.2
+ opencv-python==4.2.0.32

## Usage
Several argument that can be adjusted:
+ ``` --src-path [path]```: path of source image
+ ``` --dst-path [path]```: path of destination image
+ ``` --p [int]```: The p value for calulating weight
+ ``` --a [int]```: The a value for calulating weight
+ ``` --b [int]```: The b value for calulating weight
+ ``` --frames [int]```: Number of frames to build an animation

Command example:
```command
$ python morphing.py --src-path ./img/women.jpg --dst-path ./img/cheetah.jpg --frames 11
```

## How to use
First execute the python file.

If the images are successfully read, there will be two windows of images showing up.

Then Draw control lines on the 2 images, with the same directions and orders. Two points can make a control line.

Finished, press 'q' button and the program will start morphing.

Note that if the # of frames or pairs of control lines are too many, it will take a long time to compute.

## Reference
1. https://www.csie.ntu.edu.tw/~b97074/vfx_html/hw1.html