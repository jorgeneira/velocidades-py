from pylab import *
from scipy import ndimage 
import matplotlib.pyplot as plt
import numpy as np
	
alph=30.0
iteraciones=30
ps = (1,2,4,6,8)
qs = (1,1,5,0,2)

def escribirtxt(V,p,q):
	np.savetxt("mediciones/{}/Vx{}".format(p,q),V[:,:,0],delimiter=";",newline='\r\n')
	np.savetxt("mediciones/{}/Vy{}".format(p,q),V[:,:,1],delimiter=";",newline='\r\n')
	np.savetxt("mediciones/{}/V{}".format(p,q),V[:,:,2],delimiter=";",newline='\r\n')
	np.savetxt("mediciones/{}/Theta{}".format(p,q),V[:,:,3],delimiter=";",newline='\r\n')
	
def escribirtxt2(V,p,q):
	file = open("mediciones/{}/xyxy{}.txt".format(p,q),"w") 
	for i in range(1,shape(V)[0]-1,5):
		for j in range(1,shape(V)[1]-1,5):
				if V[i,j,2]!=0:
					file.write('{} \t {} \t {} \t {} \r\n'.format(i,j,i+V[i,j,0],j+V[i,j,1])) 
	file.close() 
	
def escribirtxt3(V,p,q):
	file = open("mediciones/{}/angmod{}.txt".format(p,q),"w") 
	for i in range(1,shape(V)[0]-1,5):
		for j in range(1,shape(V)[1]-1,5):
				if V[i,j,2]!=0:
					file.write('{} \t {} \t {} \t {} \r\n'.format(i,j,V[i,j,3],V[i,j,2])) 
	file.close() 

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray
	
def Ix(I):
	Ix= 0.25*(I[0,1,0]-I[0,0,0]+I[1,1,0]-I[1,0,0]+I[0,1,1]-I[0,0,1]+I[1,1,1]-I[1,0,1])
	return Ix
	
def Iy(I):
	Iy= 0.25*(I[1,0,0]-I[0,0,0]+I[1,1,0]-I[0,1,0]+I[1,0,1]-I[0,0,1]+I[1,1,1]-I[0,1,1])
	return Iy
	
def It(I):
	It= 0.25*(I[0,0,1]-I[0,0,0]+I[1,0,1]-I[1,0,0]+I[0,1,1]-I[0,1,0]+I[1,1,1]-I[1,1,0])
	return It
	
def baru(Vx):
	baru=(1/6)*(Vx[0,1]+Vx[1,2]+Vx[2,1]+Vx[1,0])+(1/12)*(Vx[0,0]+Vx[0,2]+Vx[2,2]+Vx[2,0])
	return baru
	
def barv(Vy):
	barv=(1/6)*(Vy[0,1]+Vy[1,2]+Vy[2,1]+Vy[1,0])+(1/12)*(Vy[0,0]+Vy[0,2]+Vy[2,2]+Vy[2,0])
	return barv
	
def u(baru,barv,Ix,Iy,It,alph):
	u=baru-Ix*(Ix*baru+Iy*barv+It)/(alph*alph+Ix*Ix+Iy*Iy)
	return u

def v(baru,barv,Ix,Iy,It,alph):
	v=barv-Iy*(Ix*baru+Iy*barv+It)/(alph*alph+Ix*Ix+Iy*Iy)
	return v
	
def main(p,q,iteraciones):
	img1=rgb2gray(imread("mediciones/{}/00{}.jpg".format(p,q),'I'))
	img2=rgb2gray(imread("mediciones/{}/00{}.jpg".format(p,q+1),'I'))
	img3=rgb2gray(imread("mediciones/{}/00{}.jpg".format(p,q+2),'I'))

	img1[img1 < 10] = 0
	img2[img2 < 10] = 0
	img3[img3 < 10] = 0

	img1=ndimage.uniform_filter(img1, size=4)
	img2=ndimage.uniform_filter(img2, size=4)
	img3=ndimage.uniform_filter(img3, size=4)

	I=np.zeros((2,2,2),float)
	Vx=np.zeros((3,3,1),float)
	Vy=np.zeros((3,3,1),float)
	dim=(shape(img1)[0],shape(img1)[1],4)
	V=np.zeros(dim)

	for k in range(iteraciones):
		for i in range(1,shape(img1)[0]-1):
			for j in range(1,shape(img1)[1]-1):
					I[0,0,0]=img2[i,j]
					I[0,0,1]=img3[i,j]
					I[0,1,0]=img2[i,j+1]
					I[0,1,1]=img3[i,j+1]
					I[1,0,0]=img2[i+1,j]
					I[1,0,1]=img3[i+1,j]
					I[1,1,0]=img2[i+1,j+1]
					I[1,1,1]=img3[i+1,j+1]
					I_x=Ix(I)
					I_y=Iy(I)
					I_t=It(I)
					Vx=V[i-1:i+2,j-1:j+2,0]
					Vy=V[i-1:i+2,j-1:j+2,1]
					bar_u=baru(Vx)
					bar_v=barv(Vy)
					V[i,j,0]=u(bar_u,bar_v,I_x,I_y,I_t,alph)
					V[i,j,1]=v(bar_u,bar_v,I_x,I_y,I_t,alph)
					if k==(iteraciones-1):
						V[i,j,2]=np.sqrt(V[i,j,1]*V[i,j,1]+V[i,j,0]*V[i,j,0])
						if V[i,j,0]!=0:
							V[i,j,3]=np.arctan2(V[i,j,1],V[i,j,0])
		print(k)
	
	for l in range(3):
		V[:,:,l]=ndimage.uniform_filter(V[:,:,l], size=3)
		V[:,:,l][V[:,:,l] < 0.06] = 0
	
	#imsave('mediciones/{}/magnitudes{}.png'.format(p,q), V[:,:,2])
	#imsave('mediciones/{}/fase{}.png'.format(p,q), V[:,:,3])
	#imsave('mediciones/{}/Vx{}.png'.format(p,q), V[:,:,0])
	#imsave('mediciones/{}/Vy{}.png'.format(p,q), V[:,:,1])

	#escribirtxt(V,p,q)
	#escribirtxt2(V,p,q)
	#escribirtxt3(V,p,q)
	
	plt.imshow(V[:,:,2])
	plt.colorbar()
	plt.show()
	
for i in range(5):
	main(ps[i],qs[i],iteraciones)