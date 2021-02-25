import cv2
import numpy as np
import matplotlib.pyplot as plt



image=np.zeros((500,500))
image[10:100,10:100]=np.eye(90)
image[300:400,200:300]=np.eye(100)

def houghline(image):
	row,cols=image.shape
	thetas=np.deg2rad(np.arange(0,180))
	length=np.ceil(np.sqrt(row**2+cols**2))
	rhos=np.linspace(-length,length,int(2*length))
	cos=np.cos(thetas)

	sin=np.sin(thetas)
	numtheta=len(thetas)
	vote=np.zeros((int(2*length),numtheta),dtype=np.uint64)

	y0,x0=np.nonzero(image)

	for i in range(len(x0)):
		x2=x0[i]
		y2=y0[i]
		for j in range(numtheta):
			rho=round(x2*cos[j]+y2*sin[j]+length)
			if isinstance(rho,int):
				vote[rho,j]+=1
			else:
				vote[int(rho),j]+=1

	c=np.where(vote>80)
	y=c[0]#[708,779]
	x=c[1]#[135,135]
	#print('y',y)
	frho=np.zeros(len(x))
	#print(frho[0])
	ftheta =np.zeros(len(x))
	k=np.zeros(len(x))
	b=np.zeros(len(x))
	x2=np.float32(np.arange(1,cols,2))
	y2=np.zeros((len(frho),len(x2)))
	
	y3=np.float32(np.arange(1,row,2))
	x3=np.zeros((len(frho),len(y3)))
	for i in range(len(x)):
		index=int(x[i])
		ftheta[i]=thetas[index]
		#print('theta',index)#135
		index2=int(y[i])
		#print('rho',index2)
		frho[i]=rhos[index2]
		if ftheta[i] ==np.pi or ftheta[i]==0:
			k[i]=0
			b[i]=frho[i]
			for j in range(len(y3)):
				x3[i][j]=np.float32(b[i])

		else:
			k[i]=-np.cos(ftheta[i])/np.sin(ftheta[i])
			b[i]=frho[i]/np.sin(ftheta[i]) 
			
			for l in range(len(x2)):
				y2[i][l]=np.float32(k[i]*x2[l]+b[i])
			
	cv2.imshow("original image",image),cv2.waitKey(0)

	for j in range(len(frho)):
		for i in range(len(x2)):
			index = int(x2[i])
			index1 = int(y2[j][i])			
			cv2.circle(image,(index,index1),3,(255,0,0),1)

	for j in range(len(frho)):
		for i in range(len(y3)):
			index2 = int(x3[j][i])
			index3 = int(y3[i])
			cv2.circle(image,(index2,index3),2,(255,0,0),1)

	cv2.imshow('hough',image)
	cv2.waitKey(0)
	

image = cv2.imread("C:\\Users\\DELL\\Desktop\\3ring.jpg")
gimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gauss = cv2.GaussianBlur(gimage, (3, 3), 0)
edges = cv2.Canny(gauss, 50, 150, apertureSize=3)
houghline(edges)

image=cv2.imread('C:\\Users\\DELL\\Desktop\\3ring.jpg')
grayimg=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gauss = cv2.GaussianBlur(grayimg, (3, 3), 0)
edges = cv2.Canny(gauss, 50, 150, apertureSize=3)
dx = cv2.Sobel(gauss, cv2.CV_16S, 1, 0) #对x求一阶导
dy = cv2.Sobel(gauss, cv2.CV_16S, 0, 1) #对y求一阶导
absX = cv2.convertScaleAbs(dx)      
absY = cv2.convertScaleAbs(dy)    
Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
#cv2.imshow('orgion',edges)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#houghspace(edges,dx,dy)

ring(Sobel,dx,dy) 