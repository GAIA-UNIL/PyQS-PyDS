from sim import runsim
import numpy
import scipy.ndimage as ndimage
import math
import os
os.environ["MKL_NUM_THREADS"] = "1"
import matplotlib.pyplot as plt

try:
	import mkl_fft as fft
except ImportError:
	try:
		import pyfftw.interfaces.numpy_fft  as fft
	except ImportError:
		import numpy.fft as fft

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
	dist=numpy.zeros(shape=kernel.shape);
	dist[math.floor(dist.shape[0]/2),math.floor(dist.shape[1]/2)]=1;
	dist=ndimage.morphology.distance_transform_edt(1-dist);
	allowedPosition=ti.copy();
	allowedPosition.flat[:]=range(allowedPosition.size);
	allowedPosition=allowedPosition[:-template.shape[0],:-template.shape[1]].flatten().astype(int);
	return runsim((ti, dist,allowedPosition, n,th,f),ti, dst,path,template,dsSample);


if __name__ == "__main__":
	
	n=25;
	th=0.005;
	f=1;

	from PIL import Image
	import requests
	from io import BytesIO

	response = requests.get('https://github.com/GAIA-UNIL/G2S/raw/master/build/TrainingImages/source.png')
	ti = numpy.array(Image.open(BytesIO(response.content)))/255;

	dst=numpy.zeros((200,200))*numpy.nan;
	kernel=numpy.ones((51,51));

	sim=ds(ti,dst,numpy.random.permutation(dst.size),kernel,n, th, f)

	plt.imshow(sim);
	plt.show();

