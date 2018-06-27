
#clear all;
#import sys
#sys.modules[__name__].__dict__.clear()

import os, time
from skimage import transform, io
import numpy as np

cwd = os.getcwd()
#path_data = './data.tif';%'Z:\gpu_heating\data.tif';
path_data = os.path.join(cwd, 'data.tif')
#inputfile_psf = './psf.tif';%'Z:\gpu_heating\psf.tif';
inputfile_psf = os.path.join(cwd, 'psf.tif')

# g = gpuDevice;
# reset(g);
# wait(g);
# g.FreeMemory;

resize_scale = 0.2
# dataA = double(ReadTifStack(path_data));
dataA = io.imread(path_data)
dataA = np.swapaxes(dataA, 0, 2)
dataA = transform.resize(dataA, np.floor(np.asarray(dataA.shape)*resize_scale))

# N = size(dataA);
N = dataA.shape

# Estimate = gpuArray(dataA);
Estimate = np.copy(dataA)
# Measure = gpuArray(dataA);
Measure = np.copy(dataA)

# FullPSF = double(ReadTifStack(inputfile_psf));
FullPSF = io.imread(inputfile_psf).astype(float)
FullPSF = np.swapaxes(FullPSF, 2, 0)
# FullPSF = FullPSF/sum(FullPSF(:));
FullPSF = FullPSF/FullPSF.sum()
FullPSF = transform.resize(FullPSF, np.floor(np.asarray(FullPSF.shape)*resize_scale))

# OTF = fftn(circshift(gpuArray(FullPSF),-floor(N/2)));
OTF = np.fft.fftn(np.roll(FullPSF, (-np.floor(np.asarray(N)/2).astype(int)).tolist()))

# iteration  = 200;
iteration = 200
# for k = 1: iteration
for k in range(iteration):

#     t_start = clock;
      t_start = time.time()
#     Blur = double(real(ifftn(fftn(Estimate).*OTF)));
      Blur = np.fft.ifftn(np.multiply(np.fft.fft(Estimate), OTF)).real
#     Diff = Measure./Blur;
      Diff = np.divide(Measure, Blur)

#     Correction = double(real(ifftn(fftn(Diff).*conj(OTF))));
      Correction = (np.fft.ifftn(np.multiply(np.fft.fftn(Diff), OTF.conj())).real).astype(float)

#     Estimate = Estimate.* Correction;
      Estimate = np.multiply(Estimate, Correction)

#     disp(['iteration # ', num2str(k), ' takes ', num2str(etime(clock,t_start)), ' s']);
      print('iteration # ' + str(k) + ' takes ' + str(time.time() - t_start) + ' s' )

#     if mod(k,40) == 0  % you can change iteration to any number to save intermediate results
      if k%40 == 0:
#     %   Estimate_Final = gather(gpu_Estimate);
#         [pathname,filename,EXT] = fileparts(path_data);
#         outputfile = strcat(pathname, '\', filename, '_iteration', num2str(k),'.tif');
          outputfile = 'data_iteration' + str(k) + '.tif'
#         WriteTifStack(gather(Estimate), outputfile, 32);  % can save output as 16 bit, if the maximal value is less than 65535.
          skimage.external.tifffile.imsave(outputfile, Estimate.astype(np.uint16))
#    end

# end
