import numpy as np 
from scipy.spatial import Delaunay 
import matplotlib.pyplot as plt 
from krig import OrdinaryKriging

 ## Hongxing Liu, Kenneth C. Jezek & Morton E. O'Kelly (2001) 
 ## Detecting outliers in irregularly distributed spatial data 
 ## sets by locally adaptive and robust statistical analysis and GIS, 
 ## International Journal of Geographical Information Science, 
 ## 15:8, 721-741, DOI: 10.1080/13658810110060442 

def getNeighborNode(ip,triangle):
	indice=np.sum(triangle==ip,axis=1).astype(bool)
	pset=[]
	for i in triangle[indice].flatten():
		if (i!=ip) and i not in pset:
			pset.append(i)
	return pset



def spatialOutlier(px,py,pz,threshold=10):
	outlier=[]
	triang=Delaunay(np.c_[px,py])
	for i in range(px.shape[0]):
		node=getNeighborNode(i,triang.simplices)
		zi=pz[node]
		wi=((px[node]-px[i])**2+(py[node]-py[i])**2)**0.5
		zhat=np.average(zi,weights=wi)		
		zexc=node.copy()
		for j in range(len(node)):
			zexc[j]=np.average(np.delete(zi,j),weights=np.delete(wi,j))

		imax=np.argmax(np.abs(zexc-zhat))
		zpred=zexc[imax]

		if abs(zpred-pz[i])> threshold:
			outlier.append(i)

	return outlier
		

def main():
	dat=np.loadtxt('../SBAS_postprocess/ASC_interseismic_pts.txt')[:,(0,1,-1)]
	# dat=np.loadtxt('../tmp2')[:,(0,1,-1)]
	dat=dat[~np.isnan(dat[:,-1])]
	x=dat[:,0]
	y=dat[:,1]


	out=spatialOutlier(x,y,dat[:,2],threshold=10)
	intx=np.linspace(min(x),max(x),100)
	inty=np.linspace(min(y),max(y),100)
	dat_filt=np.delete(dat,out,0)
	# dat_filt=dat
	# OrdinaryKriging(dat_filt[:,0],dat_filt[:,1],dat_filt[:,2],intx,inty,show_vario=True)


	# plt.triplot(dat[:,0],dat[:,1],triang.simplices)
	plt.scatter(x,y,c=dat[:,2], cmap='jet')	
	plt.plot(dat[out,0],dat[out,1],'xk')
	plt.axis('equal')
	plt.show()



if __name__ == "__main__":
	main()