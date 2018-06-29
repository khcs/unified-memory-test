
import os, time, skimage
from skimage import transform, io
import numpy as np

import cupy as cp

cwd = os.getcwd()
path_data = os.path.join(cwd, 'data.tif')
inputfile_psf = os.path.join(cwd, 'psf.tif')

resize_scale = 1.0
dataA = io.imread(path_data)
dataA = np.swapaxes(dataA, 0, 2)
dataA = transform.resize(dataA, np.floor(np.asarray(dataA.shape[:2])*resize_scale).tolist() + [dataA.shape[2]])

N = dataA.shape

Estimate = np.copy(dataA)
Measure = np.copy(dataA)

FullPSF = io.imread(inputfile_psf).astype(float)
FullPSF = np.swapaxes(FullPSF, 2, 0)
FullPSF = FullPSF/FullPSF.sum()
FullPSF = transform.resize(FullPSF, np.floor(np.asarray(FullPSF.shape[:2])*resize_scale).tolist() + [FullPSF.shape[2]])

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
