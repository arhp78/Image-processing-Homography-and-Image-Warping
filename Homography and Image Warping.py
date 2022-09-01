# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 00:03:42 2020

@author: hatam
"""
'''def findpixel(n ,m,book):
    x=int(n)
    a=n-x
    y=int(m)
    b=m-y
    res=np.zeros((1,1,3))
    res[0,0,:]=(1-b)*(1-a)*book[x,y,:]+(1-b)*(a)*book[x+1,y,:]+(b)*(1-a)*book[x,y+1,:]+(b)*(a)*book[x+1,y+1,:]
    return res
'''
import numpy as np
import cv2 
book = cv2.imread('books.jpg')
# now image1
'''
book[105:110,380:385,:]=(0,0,255)  #3
book[290:295,315:320,:]=(0,0,255)  #4
book[205:210,670:675,:]=(0,0,255)  #1
book[395:400,600:605,:]=(0,0,255)  #2
cv2.imwrite('1.jpg',book)'''
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
'''book[467:470,410:414,:]=(0,0,255)  #3
book[426:430,208:212,:]=(0,255,255)  #4
book[742:746,361:366,:]=(255,0,255)  #1
book[708:712,152:156,:]=(255,0,0)  #2
cv2.imwrite('1.jpg',book)'''
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
'''book[668:672,622:625,:]=(0,0,255)  #3
book[795:800,421:425,:]=(0,255,255)  #4
book[969:973,812:816,:]=(255,0,255)  #1
book[1100:1105,609:614,:]=(255,0,0)  #2
cv2.imwrite('1.jpg',book)'''
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

