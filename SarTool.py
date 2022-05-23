import numpy as np
from math import sin, cos, radians



def NEZ2LOS(gps,az,inc,type='ASC'):
	'''
	az: azimuth; inc: inclination; gps: ENZ order	
	az>0 ascending
	az<0 descending
	LOS extending: negative
	LOS shortening: positive
	'''
	inc=inc*3.14/180 
	az=(az-90)*3.14/180 #-90:azimuth to los
	G=[-sin(inc)*sin(az),-sin(inc)*cos(az),cos(inc)]
	print('NEZ matrix:', G)
	return np.dot(G,gps.transpose())


def LOS2EZ(ASC,DSC,azA, incA, azD, incD):
	'''LOS2EZ(IMG1, IMG2, az1, inc1, az2, inc2) -> (East,Vertical)'''
	azA=radians(azA-90)
	incA=radians(incA)
	azD=radians(azD-90)
	incD=radians(incD)
	G=[[-sin(incA)*cos(azA),cos(incA)],[-sin(incD)*cos(azD),cos(incD)]]
	print('inversion matrix=\n',np.linalg.inv(G))
	shape0=ASC.shape
	Vinv=np.dot(np.linalg.inv(G),[ASC.flatten(),DSC.flatten()])
	return  Vinv[0].reshape(shape0), Vinv[1].reshape(shape0)


def LOS2ENZ(ASC,DSC,azA, incA, azD, incD,NV):
	'''LOS2EZ(IMG1, IMG2, az1, inc1, az2, inc2) -> (East,Vertical)'''
	azA=radians(azA-90)
	incA=radians(incA)
	azD=radians(azD-90)
	incD=radians(incD)
	G=np.array([[-sin(incA)*cos(azA),cos(incA), -sin(incA)*sin(azA)],
	   [-sin(incD)*cos(azD),cos(incD), -sin(incD)*sin(azD)],
	   [0,0,1]])
	shape0=ASC.shape
	print('original matrix=\n',G)
	print('inversion matrix=\n',np.linalg.inv(G))
	Vinv=np.dot(np.linalg.inv(G),[ASC.flatten(),DSC.flatten(),NV.flatten()])
	return  Vinv[0].reshape(shape0), Vinv[1].reshape(shape0)


def SARMat(azA, incA, azD, incD):
	azA=radians(azA-90)
	incA=radians(incA)
	azD=radians(azD-90)
	incD=radians(incD)

	return np.array([[-sin(incA)*cos(azA),cos(incA),-sin(incA)*sin(azA)],
	   [-sin(incD)*cos(azD),cos(incD),-sin(incD)*sin(azD)]])


