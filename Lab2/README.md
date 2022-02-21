# Lab2 - Image Process
###### tags:`Computer Vision` 

## Introduction
In this lab, there are 3 tasks.
In task 1 - **Hybrid image**, there are many similar images pairs. Our goal is using filter to get the useful data, and blend two images to a reasonable image. 
In task 2 - **Image pyramid**, we need to construct Gaussian and Laplacian 5-layer image pyramids. This work help us  observe the residual between two pictures within different resolutions. 
In task 3 - **Colorizing the Russian Empire**, we have some Prokudin-Gorskii photo. These photos are took by special method and records negative values of R, G, B respectively. The goal is to recover the original image by aligning the red channel and the green channel to the blue channel. 

## Implementation
### Overview
#### Task 1 - Hybrid image
A hybrid image is the sum of a low-pass ﬁltered version of the one image and a high-pass ﬁltered version of a second image. There is a free parameter, which can be tuned for each image pair, which controls how much high frequency to remove from the ﬁrst image and how much low frequency to leave in the second image. 

![](https://i.imgur.com/1IX5X8O.png)


#### Task 2 - Image pyramid
An image Pyramid is a collection of representations of an image. 

![](https://i.imgur.com/s1hMo6y.png)

#### Task 3 - Colorizing the Russian Empire
Automatically produce a color image from the digitized Prokudin-Gorskii glass plate images with as few visual artifacts as possible. 

![](https://i.imgur.com/aFiEWW9.png)


### Implement detail:
#### Task 1 - Hybrid image
1. Reading image and resize:
    - Obtain the images and check whether the images are the same size. If they have different sizes, resize the smaller one to the bigger one.

2. Computing Fourier transformation and shift
    -    Multiply the input image by $(-1)^{x+y}$ to center the transform.
    -    Compute Fourier transformation of input image, i.e. F(u,v).

![](https://i.imgur.com/X7c7hMi.png)
![](https://i.imgur.com/Y3C5spK.png)

3.   Multiplying ﬁlter function
        -    Transform the original image to the low-pass filtered image by using the gaussian_filter function made by ourselves. 
        -    There is a parameter “D0”, which is called the “cutoff frequency”, controls how much frequency to leave in the transformed image and affects the blurriness of the low-pass filtered image.
        -    Then generate a high-pass filtered version of another image by subtracting a blurred version of the image itself. 
4. Creating final image and blend two images
    -    Do inverse FFT
    -    Take the real part of results as the filtered image
    -    Hybrid high-pass and low-pass images

#### Task 2 - Image pyramid
1. Reading image and resize:
    -    Read the picture and append this full-sized picture in ‘gau_pyramid’, which will store 5-layer gaussian pyramid.
2. Making other gaussian pyramid (5 layers) 
    -    Use for loop to do the gaussian blur and downsampling to get other 4 layers. 
    -    Also, collect magnitude spectrum of each pictures.
3. Making laplacian pyramid (5 layers)  
    -    Append the highest layer of the gaussian pyramid to the same layer of the laplacian pyramid
    -    From the highest layer, upsample it with nearest-neighbor interpolation, and apply the same smooth filter when constructing the gaussian pyramid.
    -    Obtain the next layer by subtracting the next layer of gaussian’s with the above result.


#### Task 3 - Colorizing the Russian Empire
1. Reading image and resize:
    -    Read in the picture data and split it to three part R, G, B for some pictures whose height cannot divided by three, we cut the border off, this will not affect the result since the border doesn’t contain useful informations. 
2. Aligning the images:
    - We use normalized cross-correlation here, and the window size is 20. Function NccAlign will give the best start index of R and G aligning to B. 


## Experimental result
#### Task 1 - Hybrid image
![](https://i.imgur.com/j5FNs2o.png)
#### Task 2 - Image pyramid
![](https://i.imgur.com/bceUBbf.png)
#### Task 3 - Colorizing the Russian Empire
![](https://i.imgur.com/CYMDmM5.png)


## Conclusion
In task 1, we know that how to blend two images together. Using two filters which are high and low, we figure out two images. We overlapped them and obtain the real part.Finally, the hybrid image is computed. In task 2, we complete gaussian blur and downsampling as well as laplacian pyramid and upsampling, they are useful in computer vision. Task 3 is a simple example of image aligning and matching, for large image, downsampling which we completed in task 2 is used
