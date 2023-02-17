# Lab4 - Structure from motion
###### tags:`Computer Vision` 

## Introduction
The goal of this homework is using two pictures to reconstruct 3D model. In homework 3, we had learned the method of finding the same points in two pictures taken in different angles. In order to find fundamental matrix, we apply 8-points algorithm on previous points we found and ratio test. After getting fundamental matrix, we can calculate 4 possible essential matrices. The final step is examining 4 possible directions of camera, and then picking the best result with maximum number of points in front of camera. 

## Implementation
#### (1) find out correspondence across images 
This step is similar to hw3. The purpose is finding the corresponding points in 2 pictures. We use cv2 build-in function to reach the goal.  
#### (2) estimate the fundamental matrix across images (normalized 8 points)  
 By step 1, we get x and x’. First we nnormorlize the points. And then apply the RANSAC with 8-points algorithm. The 
fundamental matrix is deﬁned as x’ F x = 0. 
To solve the homogeneous linear equations, we use SVD and calculate the error of F we found. And then keep the best result. 

#### (3) draw the interest points on you found in step.1 in one image  and the corresponding epipolar lines in another 
 By using fundamental matrix, the epipolar lines can be calculated as and . There are 3 parameters in l and l’, which means ax + by + c = 0. We assign x0 and y0 first, and then calculate and x1 and y1. By the two points we can draw a line.  

#### (4) get 4 possible solutions of essential matrix from fundamental matrix  
 Assuming first camera matrix is P1 = [ I | 0 ], there are four possible choices for the second camera matrix P2. First we use intrinsic matrix K to get and do SVD . 
#### (5) apply triangulation to get 3D points & find out the most appropriate solution of essential matrix 
With 4 possible camera matrix, we use triangulation to convert image points back to 3D coordinate. Next step is checking which solution will have maximum number of points in front of cameras. 
## Experimental result
![](https://i.imgur.com/kqEPEan.png)
![](https://i.imgur.com/Vwk84Hh.png)



## Conclusion
After finishing this homework, we know more about the translation of 2D and 3D coordinates. The theory about image transform can be used in many applications. Though there are a lot of packages about SfM we can use, the experience of this homework will help us a lot in the future. 