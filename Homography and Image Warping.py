# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 00:03:42 2020

@author: hatam
"""

import numpy as np
import cv2 
book = cv2.imread('books.jpg')
# now image1

pts_src=np.array([[208 ,667],[394 ,601],[107 ,383],[289 ,317]])

pts_dst=np.array([[0 ,0],[0 ,920],[1420 ,0],[1420 ,920]])
dst=np.zeros([1420,920,3])
h, status = cv2.findHomography(pts_src, pts_dst)
b=np.indices((1420,920))
b=b.reshape(2,-1)
newrow=np.ones((1,len(b[0])))
A = np.vstack([b, newrow])
dst_mat=np.matmul(np.linalg.inv(h),A)
dst_mat=dst_mat[:,:]/dst_mat[2,:]
dst_mat=np.floor(dst_mat)
dst_mat=dst_mat.astype(int)
dst=book[dst_mat[0,:],dst_mat[1,:],:]
dst=dst.reshape(1420,920,3)
'''for i in range(0,920,1):
    for j in range(0,1420,1):
        mat=np.array([[j],[i],[1]])
        res = np.dot(np.linalg.inv(h),mat) 
        if(int(res[0,0])>=95 and int(res[0,0])<=400 and int(res[1,0])<=680 and int(res[1,0])>=310):
            dst[j,i,:]=findpixel(res[0,0],res[1,0],book)
           
  '''         
cv2.imwrite("res04.jpg", dst)

# image 2
pts_src=np.array([[ 742,361],[708 ,152],[467 ,410],[426 ,208]])
pts_dst=np.array([[0 ,0],[0 ,1027],[1392 ,0],[1392 ,1027]])
dst=np.zeros([1392,1027,3])
h, status = cv2.findHomography(pts_src, pts_dst)
b=np.indices((1392,1027))
b=b.reshape(2,-1)
newrow=np.ones((1,len(b[0])))
A = np.vstack([b, newrow])
dst_mat=np.matmul(np.linalg.inv(h),A)
dst_mat=dst_mat[:,:]/dst_mat[2,:]
dst_mat=np.floor(dst_mat)
dst_mat=dst_mat.astype(int)
dst=book[dst_mat[0,:],dst_mat[1,:],:]
dst=dst.reshape(1392,1027,3)

cv2.imwrite("res05.jpg", dst)

# image 3

pts_src=np.array([[ 969,812],[1100 ,609],[668 ,622],[795 ,421]])
pts_dst=np.array([[0 ,0],[0 ,920],[1795 ,0],[1795 ,920]])
dst=np.zeros([1795,920,3])
h, status = cv2.findHomography(pts_src, pts_dst)
b=np.indices((1795,920))
b=b.reshape(2,-1)
newrow=np.ones((1,len(b[0])))
A = np.vstack([b, newrow])
dst_mat=np.matmul(np.linalg.inv(h),A)
dst_mat=dst_mat[:,:]/dst_mat[2,:]
dst_mat=np.floor(dst_mat)
dst_mat=dst_mat.astype(int)
dst=book[dst_mat[0,:],dst_mat[1,:],:]
dst=dst.reshape(1795,920,3)

cv2.imwrite("res06.jpg", dst)

