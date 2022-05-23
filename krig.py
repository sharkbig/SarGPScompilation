
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
from geopy.distance import distance,lonlat
from ArrTool import arr2nc
from scipy.optimize import curve_fit

def expr_variogram(x,y,z,bin_size=1):
	n=z.shape[0]
	dis_class=[0]
	nclass=[0]
	sqdif=[0]
	for i in range(n):
		for j in range(i+1,n):
			d=distance(lonlat(x[i],y[i]), lonlat(x[j],y[j])).km
			k=int(d//bin_size)
			while (k>len(dis_class)-1):
				dis_class+=[0]	
				nclass+=[0]
				sqdif+=[0]

			sqdif[k]+=1/2*(z[i]-z[j])**2
			dis_class[k]+=d
			nclass[k]+=1
	
	nclass=np.array(nclass)
	mask=nclass!=0
	nclass=nclass[mask]
	dis_class=np.array(dis_class)[mask]
	sqdif =np.array(sqdif)[mask]
	return dis_class/nclass,sqdif/nclass


def gaussain_model(h,w,a):
	return w*(1-np.exp(-(h/a)**2))

def power_fit(h,vario):
	m,*_ = lstsq(np.c_[np.ones(shape=h.shape),np.log(h)],np.log(vario))
	return np.exp(m[0]),m[1]

def d_vec(mx,my,x,y,func):
	n=x.shape[0]
	d=np.ones(shape=n+1)
	for i in range(n):
		dist=distance(lonlat(mx,my), lonlat(x[i],y[i])).km
		d[i]=func(dist)

	return d

def c_matrix(x,y,z,func):
	n=z.shape[0]
	c=np.zeros(shape=(n+1,n+1))
	c[:-1,-1]=1
	c[-1,:-1]=1
	for i in range(n):
		for j in range(i,n):
			if i == j: continue
			dist=distance(lonlat(x[i],y[i]), lonlat(x[j],y[j])).km
			c[i,j]=func(dist)
			c[j,i]=func(dist)
	return c


def SimpleKriging(px,py,pz,arrx,arry,func='gaussian',show_vario=False,blen=1):
	nx=arrx.shape[0]
	ny=arry.shape[0]
	intz=np.zeros(shape=(ny,nx))
	# varz=np.zeros(shape=(ny,nx))

	## compute experimental variogram
	d,gam=expr_variogram(px,py,pz,bin_size=blen)
	dtrain=d[:len(d)//2]
	gtrain=gam[:len(d)//2]
	## compute vario gram model
	if func=='power':
		w,a= power_fit(dtrain,gtrain)
		func=lambda h: w*(h**a)
	elif func=='gaussian':
		coef,cov=curve_fit(gaussain_model,dtrain,gtrain)
		func=lambda h: gaussain_model(h,*coef)
	

	if show_vario:
		plt.plot(d,gam,'o')
		plt.plot(d,func(d))
		plt.show()

	## OKG matrix
	C=c_matrix(px,py,pz,func)[:-1,:-1]
	for i in range(nx):
		for j in range(ny):
			D=d_vec(arrx[i],arry[j],px,py,func)[:-1]
			W,*_=lstsq(C,D)
			intz[j,i]=np.dot(pz,W)
	return intz


def OrdinaryKriging(px,py,pz,arrx,arry,func='gaussian',show_vario=False,blen=1):
	nx=arrx.shape[0]
	ny=arry.shape[0]
	intz=np.zeros(shape=(ny,nx))
	# varz=np.zeros(shape=(ny,nx))

	## compute experimental variogram
	d,gam=expr_variogram(px,py,pz,bin_size=blen)
	dtrain=d[:len(d)//2]
	gtrain=gam[:len(d)//2]
	## compute vario gram model
	if func=='power':
		w,a= power_fit(dtrain,gtrain)
		func=lambda h: w*(h**a)
	elif func=='gaussian':
		coef,cov=curve_fit(gaussain_model,dtrain,gtrain)
		func=lambda h: gaussain_model(h,*coef)
	

	if show_vario:
		plt.plot(d,gam,'o')
		plt.plot(d,func(d))
		plt.show()

	## OKG matrix
	C=c_matrix(px,py,pz,func)
	for i in range(nx):
		for j in range(ny):
			D=d_vec(arrx[i],arry[j],px,py,func)
			W,*_=lstsq(C,D)
			intz[j,i]=np.dot(pz,W[:-1])
	return intz


def main():
	gpsfile="/home/junyan1998/Documents/master/GPS/invert_0511.csv"
	gps=np.loadtxt(gpsfile,delimiter=',', usecols=[1,2,3,4,5])
	station_name=np.loadtxt(gpsfile,delimiter=',', usecols=[0],dtype=str)
	x=gps[:,0]
	y=gps[:,1]
	z=gps[:,2]
	sampx=25
	sampy=50
	save_name=f'N_{sampx}*{sampy}.nc'

	intx=np.linspace(min(x),max(x),sampx)
	inty=np.linspace(min(y),max(y),sampy)

	# -------------------------#
	intz=OrdinaryKriging(x,y,z,intx,inty,func='gaussian',show_vario=True)
	# arr2nc(save_name,intz,intx,inty)

if __name__ == "__main__":
	main()