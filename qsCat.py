from sim import qsCat
import numpy

n=25;
k=1.2;

from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import time

response = requests.get('https://github.com/GAIA-UNIL/G2S/raw/master/build/TrainingImages/source.png')
ti = numpy.array(Image.open(BytesIO(response.content)))/255<0.5;

dst=numpy.zeros((200,200))*numpy.nan;
kernel=numpy.ones((51,51));

start = time.time()
sim=qsCat(ti,dst,numpy.random.permutation(dst.size),kernel,n,k)
end = time.time()
print(end - start)

plt.imshow(sim);
plt.show();

