#!/bin/python -env 
from SarTool import *
from ArrTool import readnc,BilinearInterp,arr2tif,arr2nc,readgdal,mul
import argparse
import numpy as np
from krig import OrdinaryKriging
import matplotlib.pyplot as plt
from osgeo import gdal 
from scipy.interpolate import interp2d
# Rasters with GPS correction.

# exportASC=splitext(ASC)[0]+"_Align.nc"
# saveVE="SenVE_2016-2018-new.nc"
# saveVZ="SenVZ_2016-2018-new.nc"
def cmdparse():
	parser=argparse.ArgumentParser(description='Back project LOS displacement to ENZ displacement')


	parser.add_argument('IMG1',help='IMG1',type=str)
	parser.add_argument('inc1',help='incident angle of IMG1',type=float)
	parser.add_argument('az1',help='azimuth angle of IMG1 (east=0, counter-clockwise positive)', type=float)
	parser.add_argument('IMG2', help= 'IMG2', type=str)
	parser.add_argument('inc2', help='incident angle of IMG2', type=float)
	parser.add_argument('az2', help='azimuth angle of IMG2', type=float)


	parser.add_argument('--gps',help='GPS file name in csv file',type=str)
	parser.add_argument('-n',type=str,default=None,help='GPS northward velocity raster (.nc)')
	parser.add_argument('-e',type=str,default=None,help='GPS easthward velocity raster (.nc)')
	parser.add_argument('-z',type=str,default=None,help='GPS up-down velocity raster (.nc)')

	parser.add_argument('-o',help='outfile basename',type=str,dest='outname')

	return parser

def VelocityInv(mat,pv,weight=None):
	pnum=pv.shape[1]
	ps=np.empty(shape=(mat.shape[1],pnum))
	masknan=~np.any(np.isnan(pv),axis=0)
	ps[:,masknan]=np.linalg.lstsq(mat,pv[:,masknan],rcond=None)[0]
	ps[:,~masknan]=np.nan
	return ps

def main():
	DEBUG=0
	args=cmdparse().parse_args()
	## data imput
	lon1, lat1, IMG1=readnc(args.IMG1,info=False)
	lon2, lat2, IMG2=readnc(args.IMG2, info=False)	
	IMG2_align=BilinearInterp(lon2,lat2,IMG2,lon1,lat1)	
	width=lon1.shape[0]
	height=lat1.shape[0]

	if args.gps:
		gps=np.loadtxt(args.gps,delimiter=',',usecols=[1,2,3,4,5])
		print('run ordinary krignig ... ')
		print('step 1: downsample grid to acceration')
		gx=lon1[::70]
		gy=lat1[::100]
		print('step 2: run interpolation ... ')
		N=OrdinaryKriging(gps[:,0],gps[:,1],gps[:,2],gx,gy,blen=1,func='power')	
		E=OrdinaryKriging(gps[:,0],gps[:,1],gps[:,3],gx,gy,func='power')
		Z=OrdinaryKriging(gps[:,0],gps[:,1],gps[:,4],gx,gy,func='power')
		print('step 3: write to file.')
		arr2tif('gn.tif',N,gx,gy)
		arr2tif('ge.tif',E,gx,gy)
		arr2tif('gz.tif',Z,gx,gy)


	GSAR=SARMat(100,34,-100,41)
	d=np.vstack([IMG1.flatten(),IMG2_align.flatten()])

	if args.e:
		xx,yy,E=readnc(args.e)
		intp=interp2d(xx,yy,E)
		E=intp(lon1,lat1)
		GSAR=np.vstack([GSAR,[1,0,0]])
		d=np.vstack([d,E.flatten()])

	if args.z:
		xx,yy,Z=readnc(args.z)
		intp=interp2d(xx,yy,Z)
		Z=intp(lon1,lat1)
		GSAR=np.vstack([GSAR,[0,1,0]])
		d=np.vstack([d,Z.flatten()])
	
	if args.n:
		xx,yy,N=readnc(args.n)
		intp=interp2d(xx,yy,N)
		N=intp(lon1,lat1)
		GSAR=np.vstack([GSAR,[0,0,1]])
		d=np.vstack([d,N.flatten()])


	print('velocity inversion matrix:')
	print(GSAR)
	vel=VelocityInv(GSAR,d)
	e=vel[0].reshape(height,width)
	z=vel[1].reshape(height,width)


	arr2tif('join_VE.tif',e,lon1,lat1)
	arr2tif('join_VZ.tif',z,lon1,lat1)


	if DEBUG:
		Sgps=mul(GSAR,[E,Z,N])
		arr2tif('Agps.tif',Sgps[0],lon1,lat1)
		arr2tif('Dgps.tif',Sgps[1],lon1,lat1)

if __name__ == '__main__':
	main()

