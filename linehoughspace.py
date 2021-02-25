import cv2
import numpy as np
import matplotlib.pyplot as plt


image=np.zeros((500,500))
image[10,10]=1
image[20,20]=1

row,cols=image.shape
thetas=np.deg2rad(np.arange(0,180))	
length=np.ceil(np.sqrt(row**2+cols**2))
	
rhos=np.linspace(-length,length,int(2*length))
cos=np.cos(thetas)	
sin=np.sin(thetas)
numtheta=len(thetas)
	

	
y,x=np.nonzero(image)


hough=np.zeros((int(2*length),numtheta))
for i in range(len(x)):
	x2=x[i]
	y2=y[i]
	for j in range(numtheta):
		rho=round(x2*cos[j]+y2*sin[j]+length)
		theta=int(np.rad2deg(thetas[j]))
		hough[rho,theta]= 1

plt.imshow(hough,'gray')
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
	
