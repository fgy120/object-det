import cv2
import numpy as np
import matplotlib.pyplot as plt


def houghspace(image,gx,gy):
	
	minr=50
	maxr=500
	rows,cols=image.shape
	#二值化
	ret,image2 = cv2.threshold(image,120,255,cv2.THRESH_BINARY)
	plt.imshow(image2,'gray')
	plt.show()
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
	#得到不为0的像素点
	y,x=np.nonzero(image2)
	#print('y,x',y.shape)
	#累加器
	accu = np.zeros((rows,cols))

	#求梯度方向
	sin=np.zeros(len(x))
	cos=np.zeros(len(x))
	for i in range(len(x)):
		sin[i]=gy[y[i],x[i]]/image[y[i],x[i]]
		cos[i]=gx[y[i],x[i]]/image[y[i],x[i]]

	#半径循环
	for r in range(maxr):
		for i in range(len(x)):
			a = x[i]-r*cos[i]
			b = y[i]-r*sin[i]
			if a <rows and b < cols and a>=0 and b >=0:
				accu[int(a),int(b)]=1
	cv2.imshow('hough',accu)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def ring(image,gx,gy):
	minr=10
	maxr=200
	thres=20
	mindelta=200
	rows,cols=image.shape
	#二值化
	ret,image2 = cv2.threshold(image,120,255,cv2.THRESH_BINARY)
	#plt.imshow(image2,'gray')
	#plt.show()
	#cv2.waitKey(0)看
	#cv2.destroyAllWindows()
	
	#得到不为0的像素点
	y,x=np.nonzero(image2)
	
	#累加器
	accu = np.zeros((rows,cols))

	#求梯度方向
	sin=np.zeros(len(x))
	cos=np.zeros(len(x))
	for i in range(len(x)):

		sin[i]=gy[y[i],x[i]]/image[y[i],x[i]]
		cos[i]=gx[y[i],x[i]]/image[y[i],x[i]]

	#半径循环
	for r in range(maxr):
		for i in range(len(x)):

			a = x[i]-r*cos[i]
			b = y[i]-r*sin[i]
			if a <rows and b < cols and a>=0 and b >=0:
				accu[int(a),int(b)]+=1
			
	
	#idx=np.argmax(accu)
	c=np.where(accu>=thres)
	y1=c[0]#
	x1=c[1]

	p1=np.zeros(len(x1))
	p2=np.zeros(len(x1))
	for i in range(len(x1)):
		p1[i] = int(y1[i])
		p2[i] = int(x1[i])

		#cv2.circle(image,(p1[i],p2[i]),5,(255,0,0),1)
		
	rrange=np.arange(0,maxr)
	accu2=np.zeros((len(p1),maxr))
	for j in range(len(p1)):
		findp1=p1[j]
		findp2=p2[j]
		for i in range(len(x)):
			r=np.sqrt((x[i]-findp1)**2+(y[i]-findp2)**2)
			if r < maxr and r>minr:
				accu2[j,int(r)]+=1

	thres=200
	c=np.where(accu2>=thres)
	x1=c[0]
	y1=c[1]
	#print(c)
	for j in range(len(x1)):
		x2=x1[j]

		findp1=int(p1[x2])
		findp2=int(p2[x2])
		cv2.circle(image,(findp1,findp2),5,(255,0,0),1)
		for i in range(len(x1)):
			y2=y1[i]
			findr=int(rrange[y2])
			cv2.circle(image,(findp1,findp2),findr,(255,0,0),1)

	plt.imshow(image,'gray')
	plt.show()
	cv2.waitKey(0)
	cv2.destroyAllWindows()

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