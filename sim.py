import numpy
import scipy.ndimage as ndimage
import math
import os
os.environ["MKL_NUM_THREADS"] = "1" # remove fft paralelization

try:
	import mkl_fft as fft
except ImportError:
	try:
		import pyfftw.interfaces.numpy_fft  as fft
	except ImportError:
		import numpy.fft as fft


def qsSample(parameter,source,template):
	(fftim,fftim2, imSize,dist,n,k)=parameter;
	for r in range(1,numpy.max(numpy.array(template.shape))//2):
		if numpy.logical_not(numpy.isnan(source[dist<=r])).sum()>=n:
			break;
	source[dist>r]=numpy.nan;
	extendSource=numpy.pad(source,((0,imSize[0]-source.shape[0]),(0,imSize[1]-source.shape[1])),'constant', constant_values=numpy.nan);
	extendtemplate=numpy.pad(template,((0,imSize[0]-source.shape[0]),(0,imSize[1]-source.shape[1])),'constant', constant_values=0);
	mismatchMap=numpy.real( fft.ifft2( fftim2 * numpy.conj(fft.fft2(extendtemplate* numpy.nan_to_num(extendSource*0+1))) - 2 * fftim * numpy.conj(fft.fft2(extendtemplate* numpy.nan_to_num(extendSource)))));
	mismatchMap[-template.shape[0]+1:,:]=numpy.nan;
	mismatchMap[:,-template.shape[1]+1:]=numpy.nan;
	indexes=numpy.argpartition(numpy.roll(mismatchMap,tuple(x//2 for x in template.shape),(0,1)).flat,math.ceil(k));
	return indexes[int(math.floor(numpy.random.uniform(k)))];

def qs(ti,dst,path,template,n,k):
	dist=numpy.zeros(shape=template.shape);
	dist[math.floor(dist.shape[0]/2),math.floor(dist.shape[1]/2)]=1;
	dist=ndimage.morphology.distance_transform_edt(1-dist);
	return runsim((fft.fft2(ti), fft.fft2(ti**2),ti.shape, dist, n, k),ti, dst,path,template,qsSample);

def dsSample(parameter,source,template):
	(ti,dist,allowedPosition,n,th,f,)=parameter;
	for r in range(1,numpy.max(numpy.array(template.shape))//2):
		if numpy.logical_not(numpy.isnan(source[dist<=r])).sum()>=n:
			break;
	source[dist>r]=numpy.nan;
	dataLoc=numpy.where(numpy.logical_not(numpy.isnan(source)).flat);
	data=source.flat[dataLoc];
	dxL,dyL=numpy.unravel_index(dataLoc,template.shape);
	deltas=numpy.ravel_multi_index([dxL, dyL],ti.shape)
	scanPath=numpy.random.permutation(allowedPosition)[:math.ceil(ti.size*f)];

	hx=template.shape[0]//2;
	hy=template.shape[1]//2;

	bestP=numpy.random.randint(ti.size);
	if(numpy.sum(numpy.logical_not(numpy.isnan(source)))<1):
		return bestP
	bestError=numpy.inf;
	sourcelocal=numpy.zeros(source.shape);
	for p in scanPath:
		missmatch=numpy.mean((ti.flat[deltas+p]-data)**2);
		
		if(missmatch<bestError):
			bestP=p;
			bestError=missmatch;
		if(bestError<th):
			break;
	return bestP+numpy.ravel_multi_index([hx, hy],ti.shape);

def ds(ti,dst,path,template,n,th,f):
	dist=numpy.zeros(shape=template.shape);
	dist[math.floor(dist.shape[0]/2),math.floor(dist.shape[1]/2)]=1;
	dist=ndimage.morphology.distance_transform_edt(1-dist);
	allowedPosition=ti.copy();
	allowedPosition.flat[:]=range(allowedPosition.size);
	allowedPosition=allowedPosition[:-template.shape[0],:-template.shape[1]].flatten().astype(int);
	return runsim((ti, dist,allowedPosition, n,th,f),ti, dst,path,template,dsSample);


def runsim(parameter,ti,dst,path,template,sampler):
	xL,yL=numpy.unravel_index(path, dst.shape)
	hx=template.shape[0]//2;
	hy=template.shape[1]//2;
	source=template.copy();
	for x,y,p,idx in zip(xL,yL,path,range(path.size)):
		#print(x,y,p,idx)
		if(idx%(path.size//100)==0):
			print(idx*100//path.size," %");
		source*=numpy.nan;
		source[max(0,x-hx)-(x-hx):min(dst.shape[0]-1,x+hx)-x+hx+1,max(0,y-hy)-(y-hy):min(dst.shape[0]-1,y+hy)-y+hy+1]=\
			dst[max(0,x-hx):min(dst.shape[0],x+hx)+1,max(0,y-hy):min(dst.shape[1],y+hy)+1];
		
		simIndex=sampler(parameter,source,template);
		dst.flat[p]=ti.flat[simIndex];
	return dst;

	