# Lab3 - Automatic Panoramic Image Stitching
###### tags:`Computer Vision` 

## Introduction
In this work, we stitch two photos(same objects of different views) together in order to get their panoramic image. In the first, we need to find out two pictures’ interest points and feature description. So we can know the same features in two pictures. Because the two photos were taken in different views, we need to calculate the homography matrix by using RANSAC algorithm.

## Implementation
### Overview
#### (a) Interest points detection & feature description by SIFT 
We use the built-in functions in cv2 to get the interest points 
and descriptions of img1 and img2. 
#### (b) Feature matching by SIFT features 
We choose brute-force method to find two points in img2 with the minimum L2 distance between img1.And doing ratio test, if the distance of point 1 is much smaller than distance of point 2. It means it is the match point. After getting the match result, show the picture with matching 
points. 

#### (c) RANSAC to find homography matrix H
Based on RANSAC algorithm, we choose 4 match points to find homography matrix and calculate the distance between line and points. After doing above steps many times, we keep the best result to do next part. 
#### (d) Warp image to create panoramic image 
We wrap img1 onto img2 with homography matrix we got in previous step and fix some pixels by linear interpolation. 
The last step is stitch two pictures to get the panoramic
image. 

## Experimental result
(a)
![](https://i.imgur.com/rAN5xAG.jpg)
(b)
![](https://i.imgur.com/2gB32WC.png)


## Conclusion
In this work, we implement simple image stitching and also tried to run it on our own pictures. This work help us be familiar with the transformation of different views with homography matrix and find out the features of pictures. These methods in computer vision are important concepts. We believe this experience is really good and meaningful for our future works. 