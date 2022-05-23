import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from osgeo import gdal, gdalconst, osr

def readnc(name, Bandname='Band1',sample=1,info=False):
	shade=nc.Dataset(name,'r')
	if info:
		print("OPEN DATASET:",name)
		print("DATASET INFO:")
		print(shade,'\n')
	x=shade['lon'][::sample]
	y=shade['lat'][::sample]
	z=shade[Bandname][::sample,::sample]
	shade.close()
	return x,y,z.filled(np.nan)

def readgmt(sar,coord=True,show=False):
	dataset=nc.Dataset(sar,'r')
	xn,yn=dataset['dimension'][:]
	xp,yp=dataset['spacing'][:]
	xmin,xmax=dataset['x_range'][:]
	ymin,ymax=dataset['y_range'][:]
	xx=np.arange(xmin,xmax,xp)
	yy=np.arange(ymin,ymax+yp,yp)
	arr=dataset['z'][:].reshape((yn,xn))
	if show: plt.imshow(arr,cmap='jet');plt.show()
	if coord:
		return xx,yy,arr
	else:
		return arr

def readgdal(filename,sample=1):
	dataset= gdal.Open(filename, gdalconst.GA_ReadOnly)
	arr=dataset.GetRasterBand(1).ReadAsArray()[::sample,::sample]
	gtrans=dataset.GetGeoTransform()
	nx=dataset.RasterXSize
	ny=dataset.RasterYSize
	xx=(np.arange(0,nx)+0.5)*gtrans[1]+gtrans[0]
	yy=(np.arange(0,ny)+0.5)*gtrans[5]+gtrans[3]
	del dataset
	return xx[::sample],yy[::sample],arr

def arr2tif(filename,arr,lons,lats):
	if ~filename.endswith('.tif'): filename+'.tif'
	x_res=lons.shape[0]; y_res=lats.shape[0]
	dx=lons[1]-lons[0]
	dy=lats[1]-lats[0]
	x0=lons[0]-dx/2
	y0=lats[0]-dy/2
	dset=gdal.GetDriverByName('GTiff').Create(filename,x_res,y_res,1, gdal.GDT_Float32)
	dset.SetGeoTransform([x0,dx,0,y0,0,dy])
	dset.GetRasterBand(1).WriteArray(arr)
	dset.FlushCache()
	print('Write to file',filename)

def arr2nc(filename,arr,lons,lats):
	fp=nc.Dataset(filename,'w')
	# create Dimension
	dlat=fp.createDimension('lat',lats.shape[0])
	dlon=fp.createDimension('lon',lons.shape[0])
	# create Variable
	vlat=fp.createVariable('lat',lats.dtype,('lat'))
	vlon=fp.createVariable('lon',lons.dtype,('lon'))
	var=fp.createVariable('Band1',arr.dtype,('lat','lon'))
	# write
	vlat[:]=lats[:]
	vlon[:]=lons[:]
	var[:,:]= arr
	fp.close()


def smoothing(arr,typ="uniform", order=25):
	if order%2==0:order+=1
	row,col=arr.shape
	filt=np.zeros(shape=(row-order+1,col-order+1))
	for i in range(order):
		for j in range(order):
			filt+=arr[i:row-order+1+i,j:col-order+1+j]
	arr[order//2:row-order//2,order//2:col-order//2]=filt/(order**2)

	if typ=="Gaussian":
		print('Average: Gaussian; Reset order to 3')
		filt=arr[1:-1,1:-1]*0.8948+ \
		     (arr[:-2,1:-1]+arr[2:,1:-1]+arr[1:-1,2:]+arr[1:-1,:-2])*0.0256+ \
		     (arr[2:,2:]+arr[2:,:-2]+arr[:-2,2:]+arr[:-2,:-2])*0.0007
		arr[1:-1,1:-1]=filt
	if typ=="Harmony":
		for i in range(order):
			for j in range(order):
				filt+=1/arr[i:row-order+1+i,j:col-order+1+j]
		arr[order//2:row-order//2,order//2:col-order//2]=(order*order)/filt
		print(arr)
	return arr


def rastersample(pX, pY,rastX ,rastY, dist=5):
	xi=np.argmin(np.abs(rastX-pX))
	yi=np.argmin(np.abs(rastY-pY))
	yn=rastY.shape[0]
	xn=rastX.shape[0]
	exportBool=np.zeros(shape=(yn,xn),dtype=bool)
	if xi-dist<0 | yi-dist<0 | yi+dist>yn | xi+dist>xn: return exportBool
	exportBool[yi-dist:yi+dist,xi-dist:xi+dist]=True
	return exportBool




def sampleARR(gps,xcoord,ycoord,sar,plot=False):
	## sample raster from point
	## the same function as GMT::grdtrack
	output=np.empty(gps.shape[0])
	k=0
	for i in gps:
		select=sar[rastersample(i[0],i[1],xcoord,ycoord)]
		mask2= ~np.isnan(select)
		output[k]=np.average(select[mask2])
		k+=1		
	if plot:
		plt.imshow(sar,extent=[xcoord[0],xcoord[-1], np.min(ycoord),np.max(ycoord[0])], cmap='rainbow' )
		plt.scatter(gps[:,0], gps[:,1], c=output, cmap='rainbow',lw=1,edgecolors='k')
		plt.show()
	return output


def reprojection(inputfile,referencefile,outputfile,driv="GTiff"):
	print("Reprojection start ...")
	input1 = gdal.Open(inputfile, gdalconst.GA_ReadOnly)
	inputProj = input1.GetProjection()
	inputTrans = input1.GetGeoTransform()

	reference = gdal.Open(referencefile, gdalconst.GA_ReadOnly)
	referenceProj = reference.GetProjection()
	referenceTrans = reference.GetGeoTransform()
	bandreference = reference.GetRasterBand(1)
	x = reference.RasterXSize 
	y = reference.RasterYSize
	driver= gdal.GetDriverByName(driv)
	output = driver.Create(outputfile, x, y, 1, bandreference.DataType)
	output.SetGeoTransform(referenceTrans)
	output.SetProjection(referenceProj)

	gdal.ReprojectImage(input1, output, inputProj, referenceProj, gdalconst.GRA_NearestNeighbour)
	print("Reprojection Done ...\n\n")
	del output


def planefit(x,y,z,*grid):
	##### gps deramp #####
	ps=np.ones(shape=(x.shape[0],3))
	ps[:,0]=x
	ps[:,1]=y
	## inversion [x,y].*[a,b]=Z
	m=np.linalg.solve(np.dot(ps.transpose(),ps),np.dot(ps.transpose(),z))

	if len(grid)!=0:
		xy=np.transpose([grid[0],grid[1],np.ones(shape=grid[0].shape)],(1,2,0))
		return np.dot(xy,m)
	else: return m


def BilinearInterp(xcoord,ycoord,zarr,new_x,new_y):
	xmask=(new_x>=xcoord[0])*(new_x<=xcoord[-1])
	ymask=(new_y>=ycoord[0])*(new_y<=ycoord[-1])

	dx0=xcoord[1]-xcoord[0]
	dy0=ycoord[1]-ycoord[0]

	XNN=(new_x-xcoord[0])/dx0
	XNN[XNN<0]=0
	XNN[XNN>xcoord.shape[0]-1]=-1

	left=np.floor(XNN).astype(int)
	right=np.ceil(XNN).astype(int)

	d1=new_x-xcoord[left]
	d2=xcoord[right]-new_x
	dx=d2+d1
	dx[dx==0]=1
	d1[d1==d2]=1

	interp=(zarr[:,left]*d2+zarr[:,right]*d1)/dx
	# interp[:, ~xmask]=np.nan

	YNN=(new_y-ycoord[0])/dy0
	YNN[YNN>ycoord.shape[0]-1]=-1
	YNN[YNN<0]=0

	top=np.ceil(YNN).astype(int)
	bottom=np.floor(YNN).astype(int)
	d3=new_y-ycoord[bottom]
	d4=ycoord[top]-new_y
	dy=d4+d3
	dy[dy==0]=1
	d3[d3==d4]=1
	
	interp=(interp[bottom,:].T*d4 +interp[top,:].T*d3)/dy

	interp=interp.T
	# interp[~ymask,:]=np.nan

	return interp

def mul(A,B):
	if type(B) is list: B=np.array(B) 
	Bi=np.empty(shape=(A.shape[0],B.shape[1],B.shape[2]))
	for irow in range(A.shape[0]):
		Bi[irow]=np.tensordot(A[irow].reshape(1,-1),B,axes=1)
	return Bi