# Lab1 - Camera calibration

###### tags:`Computer Vision` 

## Introduction
Camera calibration is the process of estimating the parameters of a camera that produced images or video. By using these parameters, we can correct distortion, measure the size of an object in world units, or determine the location of the camera in the scene. We use this technology in applications such as machine vision (for detecting and measuring objects) or navigation systems, and 3-D scene reconstruction. 

In this lab, we need to use chessboard(set the world coordinate system to the corner of chessboard) to figure out the intrinsic matrix and extrinsic matrix.
![](https://i.imgur.com/AMl9QLO.png)

## Implementation
### Outline 
![](https://i.imgur.com/zg9d3l2.png)
### Procedure:
1. First, figure out the Hi(homography matrix) of each images.
2. Use Hi to find B ($K^{-T}K^{-1}$), and calculate intrinsic matrix K from B by using Cholesky factorization.
3. Then, get extrinsic matrix [R|t] for each images by K and H.
### Implement detail:
#### Step 1 : Calculate the Hi of each images
-    We can represent K[R t] with H and get two equations from 1 point (u,v,1)![](https://i.imgur.com/C9noOUA.png)
-    Then we use matrix mutiplication to show these two equations. Also, we can extend these formula from 1 point to N points.
![](https://i.imgur.com/rT3Qcu7.png)

-    $(X^0, Y^0) ….(X^{N-1}, Y^{N-1})$ are calculated by `cv2.findChessboardCorners`

-    $(u^0, v^0)….(u^{N-1}, v^{N-1})$, they would be like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)

- Since we have known [x y] [u v], we can get H by using SVD, below is the code how we get H.![](https://i.imgur.com/n8XSNCW.png)

#### Step 2 : Calculate the intrinsic matrix K
-    Since all points on the chessboard lie in a plane, their W  component is 0 in world coordinates we can thus delete 3rd column of extrinsic matrix![](https://i.imgur.com/GsFUdN9.png)
-   (r1, r2, r3) form an orthonormal basis, so we can get 2 equations![](https://i.imgur.com/c7vT0rg.png)

-    We set $B = K^{-T}K^{-1} = (b_{11},b_{12}, b_{13},b_{22},b_{23},b_{33})$, so B is symmetric and positive definite![](https://i.imgur.com/xxkBbmc.png)

- We use B and V to represent above 2 equation $h_i^TBh_j = v_{ij}^Tb$![](https://i.imgur.com/qW1i1w3.png)

-    V is known because of it is composed of the homography matrix. We can use SVD to get the matrix B. ![](https://i.imgur.com/fJx5V7r.png)

-   We can use Cholesky factorization to get K.![](https://i.imgur.com/CzXZfOb.png)

#### Step 3: Get extrinsic matrix [R|t] by K and H 
Now we have everything we need, so we get extrinsic matrix [R|t] for each images by K and H 
![](https://i.imgur.com/cVrrLD3.png)
![](https://i.imgur.com/xjSCjhq.png)

## Experimental result 
We run our code with two data set. One is comprised of 10 
pictures from TA, named as 0000.png~0009.jpg. The other has 10 pictures took by our member’s smartphone, named as img0.jpg-img9.jpg. 

(a). pictures provided from TA ![](https://i.imgur.com/wpGiYbN.png)

(b)  pictures we made ![](https://i.imgur.com/3fjtGYQ.png)

## Conclusion 
After this process we get the intrinsic and extrinsic matrix of our camera. We also found another way to calibrate camera, which use circular grid rather than chessboard. 
Though our intrinsic matrix is a little different from TA’s, it still works.![](https://i.imgur.com/bknSHfz.png)
With an calibrated camera, we can do 3-D scene reconstruction or other further application. 

## Reference
-    Computer Vision Ch2 PPT 
-    Course Video 
-    Demystifying Geometric Camera Calibration for Intrinsic Matrix.  https://kushalvyas.github.io/calib.html 
-    相機內外參數校正與實作 - 張正友法
https://abcd40404.github.io/2018/09/16/zhang-method/ 
-    相機標定（具體過程詳解）張正友、單應矩陣、B、R、T https://www.itread01.com/content/1546857202.html 

