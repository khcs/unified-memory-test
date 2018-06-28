
import os, time, skimage
from skimage import transform, io
import numpy as np

import cupy as cp

cwd = os.getcwd()

scale = 1024 
dataA = np.random.randn(scale, scale, 100)

N = dataA.shape

Estimate = np.copy(dataA)
Measure = np.copy(dataA)

FullPSF = np.random.randn(scale, scale, 100)

OTF = np.fft.fftn(np.roll(FullPSF, (-np.floor(np.asarray(N)/2).astype(int)).tolist()))

cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)


Estimate = cp.array(Estimate)
Measure = cp.array(Measure)
OTF = cp.array(OTF)


iteration = 200
for k in range(iteration):

      t_start = time.time()
      Blur = cp.fft.ifftn(cp.multiply(cp.fft.fft(Estimate), OTF)).real
      Diff = cp.divide(Measure, Blur)

      Correction = (cp.fft.ifftn(cp.multiply(cp.fft.fftn(Diff), OTF.conj())).real).astype(float)

      Estimate = cp.multiply(Estimate, Correction)

      print('iteration # ' + str(k) + ' takes ' + str(time.time() - t_start) + ' s' )

      if k%40 == 0:
          outputfile = 'data_iteration' + str(k) + '.tif'
          Estimate_ = cp.asnumpy(Estimate)
          skimage.external.tifffile.imsave(outputfile, Estimate_.astype(np.uint16))
