import os
import sys
import numpy as np
import PIL.Image as Image
import cv2
# import tensorflow as tf 

path='F:\\lunwen\\dataset\\gray'
# path='F:\\lunwen\\dataset\\process'
path1='F:\\lunwen\\IP102\\Detection\\VOC2007\\JPEGImages'

filenames=os.listdir(path1)
num=0
p='IP093'
for i in filenames:
	# print(i)
	# print(os.path.join(path,i))
	
	if i.split('.')[0][0:5]==p and i.split('.')[1]=='jpg':
		# #转灰度直方图均衡
		image=cv2.imread(os.path.join(path1,i))
		image=cv2.resize(image,(800,800))
		image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		# # dst=cv2.equalizeHist( grayimg )#直方图均衡
		# # print(2)
		cv2.imwrite(os.path.join('F:\\lunwen\\dataset\\gray',i),image)

	#根据json删除多余图片
	# if

	#数据增强#ano,pod,cry,rhy
	
		print(i)
		num+=1
		# image=cv2.imread(os.path.join(path,i))
		# NumPy.'img' = A single image.
		flip_1 = np.fliplr(image)
		temp = cv2.GaussianBlur(image, (5, 5), 1.5) 
		flip_vertical=cv2.flip(image,0)
		cv2.imwrite(os.path.join('F:\\lunwen\\dataset\\gray',p+'flip%s.jpg'%num),flip_1)
		cv2.imwrite(os.path.join('F:\\lunwen\\dataset\\gray',p+'verti%s.jpg'%num),flip_vertical)
		cv2.imwrite(os.path.join('F:\\lunwen\\dataset\\gray',p+'gauss%s.jpg'%num),temp)
