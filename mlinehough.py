import cv2
import numpy as np
import matplotlib.pyplot as plt



# image=np.zeros((500,500))
# image[10:100,10:100]=np.eye(90)
# image[300:400,200:300]=np.eye(100)
#rho = 1  # 线段以像素为单位的精度，double类型，1.0为1个像素
#theta = np.pi / 180  # 通过步长为1的半径和步长为π/180的角度，搜索所有可能的直线。
#threshold = 10  # 累加平面的阈值，int，超过设定阈值才被判定检测出线段，值越大，检出的线段个数越少。
#minLineLength = 20  # 能组成一条直线的最少点的数量. 点数量不足的直线将被抛弃.如果小于该值，则不被认为是一条直线.
#maxLineGap = 10#断定为一条线段的最大容许间隔（断裂）.直线间隙最大值，如果两条直线间隙大于该值，则被认为是两条线段，否则是一条。
 
#lines = cv2.HoughLinesP(edges, rho, theta, threshold,
 #                           minLineLength, maxLineGap)
def houghline(image):
	row,cols=image.shape
	thetas=np.deg2rad(np.arange(0,180))#角度转弧度（创建步长默认为1的0到179数组）
	#print('thetas',thetas)
	length=np.ceil(np.sqrt(row**2+cols**2))#向上取整(得到对角线长)708
	#print('length',length)
	rhos=np.linspace(0,length,int(length))#numpy.linspace(start, stop, num， endpoint, retstep, dtype, axis)在start和stop之间返回均匀间隔的数据start:返回样本数据开始点
	#stop:返回样本数据结束点num:生成的样本数据量，默认为50endpoint：True则包含stop；False则不包含stop
	#retstep：If True, return (samples, step), where step is the spacing between samples.(即如果为True则结果会给出数据间隔)
	#dtype：输出数组类型axis：0(默认)或-1
	#print('rhos',rhos)
	cos=np.cos(thetas)
	#print('cos',cos)
	sin=np.sin(thetas)
	numtheta=len(thetas)#列表元素数目或字符串长度180
	vote=np.zeros((int(length),numtheta),dtype=np.uint64)
	#print('numtheta',numtheta)
	y,x=np.nonzero(image)#二维数组时，分别从row，col得到不为0的索引
	#print('y,x',y)
	#print(x)

	for i in range(len(x)):
		x2=x[i]
		y2=y[i]
		for j in range(numtheta):
			rho=round(x2*cos[j]+y2*sin[j])
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

	for i in range(len(x)):
		index=int(x[i])
		ftheta[i]=thetas[index]
		#print('theta',index)#135
		index2=int(y[i])
		#print('rho',index2)
		frho[i]=rhos[index2]
		k[i]=-np.cos(ftheta[i])/(np.sin(ftheta[i])+1)#转换为y=kx+b形式 -0.9999999
		b[i]=frho[i]/(np.sin(ftheta[i])+1) #0.7076065032933097
		#x2=np.float32(np.arange(1,300,2))
		#y2=np.zeros((len(frho),len(x2)))
		for l in range(len(x2)):
			y2[i][l]=np.float32(k[i]*x2[l]+b[i])
			#print('y2',int(y2[i][l]))
		#print('y2',int(y2[0][30]))



	cv2.imshow("original image",image),cv2.waitKey(0)

	for j in range(len(frho)):
		for i in range(len(x2)):
			index = int(x2[i])
			index1 = int(y2[j][i])
			#print('y2',int(y2[j][i]))
			cv2.circle(image,(index,index1),3,(255,0,0),1)
			#print('x,y',int(x2[i]),int(y2[j][i]))


	cv2.imshow('hough',image)
	cv2.waitKey(0)
	#for i in range(len(frho)):
		#print ("rho={0:.2f}, theta={1:.0f}".format(frho[i], np.rad2deg(ftheta[i])))

	#cv2.waitKey(0)
	cv2.destroyAllWindows()

img = cv2.imread("C:\\Users\\DELL\\Desktop\\2.jpg")
gimage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gauss = cv2.GaussianBlur(gimage, (3, 3), 0)
edges = cv2.Canny(gauss, 50, 150, apertureSize=3)
houghline(edges) 