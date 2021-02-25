#-*- coding:utf-8 –*-
import os 
import cv2
from shutil import copy
#png转JPG
path='F:\\lunwen\\code\\MyData\\pest_dete_reco\\data\\sourcedata\\黑颜蝇'
flie=os.listdir(path)
for f in flie:
	if f.split('.')[1]=='png':
		imgname=f.split('.')[0]+'.jpg'
		copy(os.path.join(path,f),os.path.join(path,imgname))

#删除指定类型图片
file_name = path
for root, dirs, files in os.walk(file_name):
	for name in files:
		if name.endswith(".png"): # 填写规则
			os.remove(os.path.join(root, name))
			print("Delete File: " + os.path.join(root, name))