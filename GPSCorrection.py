from os.path import splitext
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
from scipy.interpolate import griddata 
from ArrTool import *
import SarTool as Tools
from krig import OrdinaryKriging, SimpleKriging


def input():
	parser=argparse.ArgumentParser(description='SAR-GPS correction')
	parser.add_argument('SAR',help='path to SAR image',type=str)
	parser.add_argument('GPS',help='GPS and Sanpled SAR data by GMT grdtrack',type=str)
	parser.add_argument('inc',help='inclination angle',type=float)
	parser.add_argument('az',help='azimuth angle', type=float)
	parser.add_argument('-ref',help='refence GPS station')
	parser.add_argument('--outfile',type=str,help='basename of the file output',default=False)
	parser.add_argument('-d','--debug',action='count',help='DEBUG mode',dest='DEBUG')
	parser.add_argument('-m','--method',default='KG',dest='itp_method',
							help='interpolation method for residual [KG/BL]')
	return parser


def statistic_report(x,y):
	c=np.polyfit(x,y,1)
	R2=c[0]*np.std(x)/np.std(y)
	print(f'y={c[1]:.4f}+{c[0]:.4f}*x')
	print('R2:',R2)
	fout=lambda x: c[1]+c[0]*x
	return fout


def deramp(x,y,res,gridx,gridy):
	n=x.shape[0]
	G=np.c_[np.ones(shape=n),x,y,x*y,x**2,y**2]
	c0,cx,cy,cxy,cxx,cyy=linalg.lstsq(G,res)[0]
	ramp=cx*gridx+cy*gridy+c0+cxy*gridx*gridy+cxx*gridx**2+cyy*gridy**2
	rfit=cx*x+cy*y+c0+cxy*x*y+cxx*x**2+cyy*y**2
	return ramp,rfit
	
def main():
	###### parameter #######
	sample=1

	# SAR="/home/junyan1998/Documents/master/SBAS/Sen/2016-2018/SenDsc-2016-2018-new/SenDSC-2016-2018-new.grd"
	# SAR="/home/junyan1998/Documents/master/SBAS/Sen/SenDsc-2018-2020/SbasDSC2018-2020.tif"
	var=input().parse_args()
	SAR=var.SAR
	vfile=var.GPS
	inc=var.inc
	az=var.az
	GPSDEBUG=var.DEBUG
	itp_method=var.itp_method
	if var.outfile:
		basename=var.outfile
	else:
		basename=splitext(SAR)[0]

	######### Load DATA #########
	if SAR.endswith("nc"):
		lon, lat, los= readnc(SAR,sample=sample)
	else:
		lon,lat,los=readgdal(SAR,sample=sample)
	if lat.shape[0] != los.shape[0]: raise IndexError('latitude not compatible %i:%i'%(lat.shape[0],los.shape[0]))
	if lon.shape[0] != los.shape[1]: raise IndexError('latitude not compatible %i:%i'%(lat.shape[0],los.shape[1]))

	xx,yy=np.meshgrid(lon,lat)
	gxy=np.loadtxt(vfile,usecols=[0,1])
	vel0=np.loadtxt(vfile,usecols=[2,3,4]) #read in NEZ order(WARNING)
	sample=np.loadtxt(vfile,usecols=[5])

	mask=~np.isnan(sample)
	drop=vel0[~mask]

	#############################

	print('##### ENZ tranformation ######')
	GPS2LOS=Tools.NEZ2LOS(vel0,az,inc)
	gps_pt=np.empty(shape=(vel0.shape[0],5))[mask]
	gps_pt[:,:2]=gxy[mask]
	gps_pt[:,2]=GPS2LOS[mask]
	gps_pt[:,3]=sample[mask]


	########## reference point ############
	# gps_pt=gps_pt[gps_pt[:,5]==1] 
	print('\n##### RAW Fitting #####')
	gps_mean=np.average(gps_pt[:,2])
	insar_mean=np.average(gps_pt[:,3])
	print('GPS mean:',gps_mean)
	print('InSAR mean:', insar_mean)
	residual=gps_pt[:,3]-gps_pt[:,2]

	predict_sar=statistic_report(gps_pt[:,2],gps_pt[:,3])
	if GPSDEBUG==1:
		GPSPLOT(gps_pt[:,2],gps_pt[:,3],predict_sar)
	


	#####################################
	print('\n###### GPS deramp ######')
	ramp,rfit=deramp(gps_pt[:,0],gps_pt[:,1],residual,xx,yy)
	gps_pt[:,4]=gps_pt[:,3]-rfit
	los_ramp=los-ramp
	res_deramp=gps_pt[:,4]-gps_pt[:,2] 

	# report 
	func2=statistic_report(gps_pt[:,2],gps_pt[:,4])
	if GPSDEBUG==1:
		GPSPLOT(gps_pt[:,2],gps_pt[:,4],func2)

	arr2nc(basename+'_deramp.nc',los_ramp,lon,lat)
	arr2nc(basename+'_residual_ramp.nc',ramp,lon,lat )


	####################################

	if itp_method == "KG":
		print('\n##### Kriging residual correction ######')
		xsamp=lon[::100]
		ysamp=lat[::100]
		px,py=np.meshgrid(xsamp,ysamp)

		from outlierDetect import spatialOutlier
		outlier=spatialOutlier(gps_pt[:,0],gps_pt[:,1],res_deramp,threshold=3)
		print('exclude outlier point for kriging interpolate:',outlier)
		print('filtering threshold (mm/yr):',3)
		mask=np.ones(shape=gps_pt.shape[0],dtype=bool)
		mask[outlier]=False
		point=np.c_[px.flatten(),py.flatten()]
		if GPSDEBUG==2:
			plt.scatter(gps_pt[:,0],gps_pt[:,1],c=res_deramp,cmap='jet')
			plt.show()



		R_OK=SimpleKriging(gps_pt[mask,0],gps_pt[mask,1],res_deramp[mask],xsamp,ysamp,func='power',show_vario=(GPSDEBUG==2),blen=1)
		
		res_cor=griddata(point,R_OK.flatten(),(xx,yy),method='linear')
		los_ramp_cor=los_ramp-res_cor
		
	elif itp_method == 'BL':
		print('\n##### Bilinear residual correction #####')
		res_cor=griddata(gps_pt[:,:2],res_deramp,(xx,yy),method='linear')
		res_ext=griddata(gps_pt[:,:2],res_deramp,(xx,yy),method='nearest')
		res_cor[np.isnan(res_cor)]=res_ext[np.isnan(res_cor)]
		los_ramp_cor=los_ramp-res_cor

	elif itp_method == 'False':
		print('No residual correction applied.')

	else:
		print('unknown method')
		

	arr2nc(f'{basename}_deramp_{itp_method}.nc',los_ramp_cor, lon, lat)
	arr2nc(f'{basename}_residual_{itp_method}.nc',res_cor,lon,lat)
	np.savetxt(basename+'_pts.txt',gps_pt)
	print('write to gps_ps: x,y, gps(los),sar(original),sar(deramp)')
	

def GPSPLOT(gps,sar,func):
		print("GPS DEBUG plot ... \n")
		plt.figure(figsize=(14,7))
		plt.subplot(121)
		plt.scatter(gps,sar, cmap="seismic")
		plt.clim(-5,5)
		s=np.linspace(plt.xlim()[0],plt.xlim()[1])

		## plot regression line
		plt.plot(s,func(s))
		plt.xlabel('GPS at LOS')
		plt.ylabel('InSAR at LOS')
		plt.axis('equal')
		plt.subplot(122)
		plt.hist(gps-sar,bins=20)
		ymax=plt.ylim()[1]
		# plt.plot([stdr,stdr],[0,20],'k--')
		# plt.plot([-stdr,-stdr],[0,20],'k--')
		plt.ylim(0,ymax)
		plt.xlabel('Residual (mm/yr)')
		plt.show()


if __name__ == "__main__":
	main()
