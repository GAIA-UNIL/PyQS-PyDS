from sim import ds
import numpy

n=25;
th=0.005;
f=1;

from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import time

response = requests.get('https://github.com/GAIA-UNIL/G2S/raw/master/build/TrainingImages/source.png')
ti = numpy.array(Image.open(BytesIO(response.content)))/255;

dst=numpy.zeros((200,200))*numpy.nan;
kernel=numpy.ones((51,51));

start = time.time()
sim=ds(ti,dst,numpy.random.permutation(dst.size),kernel,n, th, f)
end = time.time()
print(end - start)

plt.imshow(sim);
plt.show();

