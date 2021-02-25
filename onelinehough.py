import cv2
import numpy as np

img = cv2.imread("C:\\Users\\DELL\\Desktop\\4.jpg")
gimage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gauss = cv2.GaussianBlur(gimage, (3, 3), 0)
edges = cv2.Canny(gauss, 50, 150, apertureSize=3)
 

image=np.zeros((500,500))
image[10:100,10:100]=np.eye(90)
image[300:400,200:300]=np.eye(100)
#rho = 1  # 线段以像素为单位的精度，double类型，1.0为1个像素
#theta = np.pi / 180  # 通过步长为1的半径和步长为π/180的角度，搜索所有可能的直线。
#threshold = 10  # 累加平面的阈值，int，超过设定阈值才被判定检测出线段，值越大，检出的线段个数越少。
#minLineLength = 20  # 能组成一条直线的最少点的数量. 点数量不足的直线将被抛弃.如果小于该值，则不被认为是一条直线.
#maxLineGap = 10#断定为一条线段的最大容许间隔（断裂）.直线间隙最大值，如果两条直线间隙大于该值，则被认为是两条线段，否则是一条。
 
#lines = cv2.HoughLinesP(edges, rho, theta, threshold,
 #                           minLineLength, maxLineGap)

row,cols=image.shape
thetas=np.deg2rad(np.arange(0,180))#角度转弧度（创建步长默认为1的0到179数组）
#print('thetas',thetas)
length=np.ceil(np.sqrt(row**2+cols**2))#向上取整(得到对角线长)708
#print('length',length)
rhos=np.linspace(-length,length,int(2*length))#numpy.linspace(start, stop, num， endpoint, retstep, dtype, axis)在start和stop之间返回均匀间隔的数据start:返回样本数据开始点
#stop:返回样本数据结束点num:生成的样本数据量，默认为50endpoint：True则包含stop；False则不包含stop
#retstep：If True, return (samples, step), where step is the spacing between samples.(即如果为True则结果会给出数据间隔)
#dtype：输出数组类型axis：0(默认)或-1
#print('rhos',rhos)
cos=np.cos(thetas)
#print('cos',cos)
sin=np.sin(thetas)
numtheta=len(thetas)#列表元素数目或字符串长度180
#print('numtheta',numtheta)
vote=np.zeros((int(2*length),numtheta),dtype=np.uint64)
y,x=np.nonzero(image)#二维数组时，分别从row，col得到不为0的索引
#print('y,x',y)
#print(x)

for i in range(len(x)):
	x2=x[i]
	y2=y[i]
	for j in range(numtheta):
		rho=round(x2*cos[j]+y2*sin[j])+length
		if isinstance(rho,int):
			vote[rho,j]+=1
		else:
			vote[int(rho),j]+=1

idx=np.argmax(vote)#得到最大值的索引


#print('idx,',idx)#127575
rho = rhos[int(idx/vote.shape[1])]
#print('vote.shape1',vote.shape[1])#180
#print('vote.shape0',vote.shape[0])#1416
#print('rho',rho)#0.50003533568904004
theta = thetas[idx % vote.shape[1]]
#print('theta',theta)#2.356194490192345
k=-np.cos(theta)/np.sin(theta)#转换为y=kx+b形式 -0.9999999
b=rho/np.sin(theta) #0.7076065032933097
t=np.float32(np.arange(1,150,2))
#要在image 上画必须用float32，要不然会报错(float不行)
print('t',t)
r=np.float32(k*x+b)
print('r',r)
#print('k,b,x,y',k,b,x,y)
#print('lenx',len(x))#75
cv2.imshow("original image",image),cv2.waitKey(0)

for i in range(len(t)-1):
    cv2.circle(image,(t[i],r[i]),5,(255,0,0),1)
    #print('yi',y[i])
cv2.imshow("hough",image),cv2.waitKey(0)
print ("rho={0:.2f}, theta={1:.0f}".format(rho, np.rad2deg(theta)))

cv2.waitKey(0)
cv2.destroyAllWindows()