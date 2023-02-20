# Lab5 - Builds a classifier to categorize images into one of 15 scene types
###### tags:`Computer Vision` 

## Introduction
In this homework, we try to build a classifier to categorize images into one of 15 scene types. We tried different methods from task 1 to task 3 to get better results. In task 1, we implemented tiny images representation, then brute force method to find k-nearest neighbor and decide the type of test image. In task 2, we try the combination of bag of SIFT representation and nearest neighbor classifier, the accuracy of classification rise to about 50%. In task 3, nearest neighbor classifier is replaced with  linear SVM classifier. 

## Implementation
### Task 1 - Tiny images representation + nearest neighbor classiﬁer
-    resize the picture as 16*16   
-    Use k nearest neighbor algorithm, it means that given a test instance i, find the k closest neighbors and their labels. Predict i’s label as the majority of the labels of the k nearest neighbors 
### Task 2 - Bag of SIFT representation + nearest neighbor classiﬁer
-  find SIFT keypoints of all train pictures
-   function k_means(data, k, threas) build cluster of all train key points, we use k=34, threas = 1000 in while the error is bigger than threshold, we repeatedly classify all points into k clusters and compute new center of each cluster. task 2 it returns k centers of k clusters.
-   build histogram of train pictures
-   for test data, find SIFT key points and build the histogram 
-    use function ​ GetNeighbors(train, test_row, num_neighbors) in task 1, find k nearest neighbors of each test picture and classify them. 
### Task3:Bag of SIFT representation + linear SVM classiﬁer
-    use the train_histogram got in step2 with support vector machine algorithm, The objective of the svm algorithm is to find a hyperplane in an N-dimensional space that distinctly classifies the data points. In this step, we use libsvm package to simply the process. First we create the data in specified type.
-     Use grid.py in libsvm to find best parameters 
-   Train model with best parameters 


## Experimental result
### Task 1: 
k=1~10 accuracy 
![](https://i.imgur.com/kFW4KNA.png)
line chart of different k, we get best accuracy when k = 2 
![](https://i.imgur.com/9tDsHMF.png)
### Task 2: 
![](https://i.imgur.com/QvT7P3q.png)
### Task 3: 
![](https://i.imgur.com/pGlrWuL.png)



## Conclusion
After finishing this homework, we know more about the model in machine learning. There are a lot of theory in computer vision can be used in many applications. The algorithms about classifier tocategorize images and the experience of this homework will help us a lot in the future.  