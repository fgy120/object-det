# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt


def togray(image):#灰度
	rows=image.shape[0]
	cols=image.shape[1]
	gray= np.zeros((rows,cols),dtype=np.int16) 
	
	for i in range(len(image)):
		for j in range(len(image[0])):
			b=image[i,j,0]
			g=image[i,j,1]
			r=image[i,j,2]
			t=int(0.114*b+0.587*g+0.299*r)
			gray[i,j]=t
	return gray

def gauss(image):#高斯滤波
	rows=image.shape[0]
	cols=image.shape[1]
	ksize3=np.array([[0.05,0.1,0.05],[0.1,0.4,0.1],[0.05,0.1,0.05]])
	#print(ksize3[0,0])
	gauss= np.zeros((rows,cols),dtype=np.int16)
	
	image2=np.zeros((rows+2,cols+2),dtype=np.int16)
	su=0
	if len(image.shape)!=3:
		for i in range(rows):
			for j in range(cols):
				g=image[i,j]
				image2[i+1,j+1]=g
		for i in range(rows):
			for j in range(cols):
				a0=ksize3[0,0]*(image2[i,j]+image2[i,j+2]+image2[i+2,j]+image2[i+2,j+2])
				a1=ksize3[0,1]*(image2[i,j+1]+image2[i+1,j]+image2[i+1,j+2]+image2[i+2,j+1])
				a2=ksize3[1,1]*image2[i+1,j+1]
				su=a0+a1+a2
				gauss[i,j]=su
	gauss=cv2.convertScaleAbs(gauss)
	return gauss

def histaverage(image):#直方图均衡
	rows=image.shape[0]
	cols=image.shape[1]
	n=rows*cols
	# ideal=n/256
	hist=np.zeros((rows,cols),dtype=np.int16)
	k=np.zeros(256)#每个像素的累积数目
	p=np.zeros(256)#每个像素的概率
	s=np.zeros(256)#累积概率，值
	su=np.zeros(256)
	for i in range(rows):
		for j in range(cols):
			a=image[i,j]
			# for t in range(256):
			# 	if a==t:
					# k[t]+=1
			k[a]+=1
	for i in range(256):
		p[i]=k[i]/n
		for j in range(i+1):
			s[i]+=p[j]
	for i in range(256):
		s[i]=round(255*s[i])
	for i in range(rows):
		for j in range(cols):
			l=int(image[i,j])
			hist[i,j]=s[l]
	hist=cv2.convertScaleAbs(hist)
	return hist

def fushi(image,size):#腐蚀
	rows=image.shape[0]
	cols=image.shape[1]
	fushi= np.zeros((rows,cols),dtype=np.int16)
	image2=np.ones((rows+size-1,cols+size-1),dtype=np.int16)
	for i in range(int(size/2)):
		image2[0+i,0:cols+size-2]=255
		image2[0:cols+size-2,0+i]=255
		image2[rows+int(size/2)+i,0:cols+size-2]=255
		image2[0:cols+size-2,cols+int(size/2)+i]
	ksize3=np.ones((size,size))

	if len(image.shape)!=3:
		for i in range(rows):
			for j in range(cols):
				g=image[i,j]
				image2[i+int(size/2),j+int(size/2)]=g

	for i in range(rows):
		for j in range(cols):
			minp=image2[i,j]
			for k in range(size):
				for l in range(size):
					if minp>image2[i+k,j+l]:
						minp=image2[i+k,j+l]
					if ksize3[k,l]*minp==0:
						fushi[i,j]=0
						#print(i,j)
					else:
						fushi[i,j]=minp
	fushi=cv2.convertScaleAbs(fushi)
	return fushi

def penzhang(image,size):#膨胀
	rows=image.shape[0]
	cols=image.shape[1]

	penzhang= np.zeros((rows,cols),dtype=np.int16)
	image2=np.ones((rows+size-1,cols+size-1),dtype=np.int16)

	ksize3=np.ones((size,size))

	if len(image.shape)!=3:
		for i in range(rows):
			for j in range(cols):
				g=image[i,j]
				image2[i+int(size/2),j+int(size/2)]=g
	for i in range(rows):
		for j in range(cols):
			maxp=image2[i,j]
			for k in range(size):
				for l in range(size):
					if maxp < image2[i+k,j+l]:
						maxp = image2[i+k,j+l]
					if ksize3[k,l]*maxp==255:
						penzhang[i,j]=255
						#print(i,j)
					else:
						penzhang[i,j]=maxp
	penzhang=cv2.convertScaleAbs(penzhang)
	return penzhang

def kai(image,size):#开运算
	kai1=fushi(image,size)
	kai2=penzhang(kai1,size)
	return kai2

def bi(image,size):#闭运算
	bi1=penzhang(image,size)
	bi2=fushi(bi1,size)
	return bi2

def fourier(image):#傅里叶变换
	rows=image.shape[0]
	cols=image.shape[1]
	image2=np.zeros((rows,cols),dtype=np.complex128)
	mn=np.sqrt(1/(rows*cols))
	su=0
	for i in range(rows):
		for k in range(cols):
			for m in range(rows):
				for n in range(cols):
					a=2*np.pi*(i*m/rows+k*n/cols)
					su+=mn*image[m,n]*(complex(np.cos(a),-np.sin(a)))
			image2[i,k]=su

			#image2[i,k]=mn*su*cmath.exp(complex(0,-2*np.pi*(i*m/rows+k*n/cols)))
	print('fourierzj',image2)
	return fourier

image=cv2.imread('F:\\lunwen\\dataset\\all\\IP074000000.jpg')
cv2.imshow('image',image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#调用函数的灰度
grayimg=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow('grayimg',grayimg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#自己写的灰度
#gray=togray(image)
# plt.imshow(gray,'gray')
# plt.show()
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#调用函数的高斯
# gaussimage = cv2.GaussianBlur(grayimg, (3, 3), 0)
# print('gaussd',gaussimage)
# cv2.imshow('gaussd',gaussimage)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #自己写的高斯
# gauss=gauss(grayimg)
# cv2.imshow('gaussx',gauss)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#调用直方图
dst=cv2.equalizeHist( grayimg )#均衡
array2=dst.ravel()
cv2.imshow('histd',dst)
# plt.hist(array2,256)
# plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

#直方图均衡
# hist=histaverage(grayimg)
# cv2.imshow('histx',hist)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# array3 = hist.ravel()
# plt.hist(array3,256)
# plt.show()

# #调用腐蚀
# kernel = np.ones((3,3), np.uint8)
# erosion = cv2.erode(grayimg, kernel)
# cv2.imshow('fushid',erosion)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #自己写的腐蚀
# size=3
# fushi=fushi(grayimg,size)
# cv2.imshow('fushix',fushi)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #调用膨胀
# kernel= np.ones((5,5),np.uint8)
# dst = cv2.dilate(grayimg, kernel)
# cv2.imshow('penzhang',dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #自己写的膨胀
# size=5
# penzhang=penzhang(grayimg,size)
# cv2.imshow('penzhang',penzhang)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#调用开运算
# kernel = np.ones((3,3), np.uint8)
# erosion = cv2.erode(grayimg, kernel)
# dst = cv2.dilate(erosion, kernel)
# cv2.imshow('kaid',dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# #自己写的开运算
# size=3
# kai=kai(grayimg,size)
# cv2.imshow('kaix',kai)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#调用闭运算
# kernel = np.ones((3,3), np.uint8)
# dst = cv2.dilate(grayimg, kernel)
# erosion = cv2.erode(dst, kernel)
# cv2.imshow('bi',erosion)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#自己写的闭运算
# size=3
# bi=bi(grayimg,size)
# cv2.imshow('bix',bi)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#调用fourier变换
# fft2 = np.fft.fft2(grayimg)
# plt.imshow(np.abs(fft2),'gray'),plt.title('fft2')
# print('fft2',fft2)
# #自己写的
# fourier = fourier(grayimg)