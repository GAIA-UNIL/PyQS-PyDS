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


def qsSample(parameter,source,template):
	(fftim,fftim2, imSize,dist,n,k)=parameter;
	for r in range(1,numpy.max(numpy.array(template.shape))//2):
		if numpy.logical_not(numpy.isnan(source[dist<=r])).sum()>=n:
			break;
	source[dist>r]=numpy.nan;
	extendSource=numpy.pad(source,((0,imSize[0]-source.shape[0]),(0,imSize[1]-source.shape[1])),'constant', constant_values=numpy.nan);
	extendtemplate=numpy.pad(template,((0,imSize[0]-source.shape[0]),(0,imSize[1]-source.shape[1])),'constant', constant_values=0);
	mismatchMap=numpy.real( fft.ifft2( fftim2 * numpy.conj(fft.fft2(extendtemplate* numpy.nan_to_num(extendSource*0+1))) - 2 * fftim * numpy.conj(fft.fft2(extendtemplate* numpy.nan_to_num(extendSource)))));
	mismatchMap[-kernel.shape[0]+1:,:]=numpy.nan;
	mismatchMap[:,-kernel.shape[1]+1:]=numpy.nan;
	indexes=numpy.argpartition(numpy.roll(mismatchMap,tuple(x//2 for x in template.shape),(0,1)).flat,math.ceil(k));
	return indexes[int(math.floor(numpy.random.uniform(k)))];

def qs(ti,dst,path,template,n,k):
	dist=numpy.zeros(shape=kernel.shape);
	dist[math.floor(dist.shape[0]/2),math.floor(dist.shape[1]/2)]=1;
	dist=ndimage.morphology.distance_transform_edt(1-dist);
	return runsim((fft.fft2(ti), fft.fft2(ti**2),ti.shape, dist, n, k),ti, dst,path,template,qsSample);


if __name__ == "__main__":
	
	n=25;
	k=1.2;

	from PIL import Image
	import requests
	from io import BytesIO

	response = requests.get('https://github.com/GAIA-UNIL/G2S/raw/master/build/TrainingImages/source.png')
	ti = numpy.array(Image.open(BytesIO(response.content)))/255;


	dst=numpy.zeros((200,200))*numpy.nan;
	kernel=numpy.ones((51,51));

	sim=qs(ti,dst,numpy.random.permutation(dst.size),kernel,n,k)

	plt.imshow(sim);
	plt.show();

	# plt.figure()
	# plt.imshow(ti);
	# plt.show();

