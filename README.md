# Image-Morphing
Implementation of image morphing based on the paper: [Feature-Based Image Metamorphosis](https://www.cs.princeton.edu/courses/archive/fall00/cs426/papers/beier92.pdf)

This work morphs the given two images and produce a short animation from src img to dst img.

## Requirement
### C++
+ C++11
+ OpenCV==4.3.0
### Python
This work is done using python 3.6.
+ numpy==1.18.2
+ opencv-python==4.2.0.32

## How to run
### C++
Compilation:

ex:
```command
$ g++ parallel_morphing.cpp -std=c++11 -o parallel_morphing `pkg-config opencv4 --cflags` `pkg-config opencv4 --libs`
```
Several argument that can be adjusted:
+ ``` [src path]```: path of source image
+ ``` [dst path]```: path of destination image
+ ``` -p=[int]```: The p value for calulating weight
+ ``` -a=[int]```: The a value for calulating weight
+ ``` -b=[int]```: The b value for calulating weight
+ ``` -f=[int]```: Number of frames to build an animation
ex:
```command
$ ./parallel_morphing women.jpg cheetah.jpg
```
or
```command
$ ./parallel_morphing -f=51 women.jpg cheetah.jpg -p=1
```

### Python
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

## How to use the program
First execute the python file.

If the images are successfully read, there will be two windows of images showing up.

Then Draw control lines on the 2 images, with the same directions and orders. Two points can make a control line.

Finished, press 'q' button and the program will start morphing.

Note that if the # of frames or pairs of control lines are too many, it will take a long time to compute.

## P.S. 
+ ```parallel_morphing.cpp``` uses ```parallel_for_``` from OpenCV to accelerate computing.

## Reference
1. https://www.csie.ntu.edu.tw/~b97074/vfx_html/hw1.html