#-*- coding:utf-8 –*-
import os
from shutil import copy

srcpath='F:\\lunwen\\dataset\\gray'
detpath='F:\\lunwen\\code\\MyData\\pest_dete_reco\\data\\sourcedata\\柑桔潜叶蛾'

img_list = os.listdir(detpath)

for i in img_list:
	name = i.split('.')[0]
	new_obj_name = name+'.jpg'    
	copy(srcpath+'\\'+new_obj_name, detpath+'\\'+new_obj_name)