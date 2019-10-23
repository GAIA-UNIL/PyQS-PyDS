import numpy

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