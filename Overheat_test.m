clear all;

path_data = './data.tif';%'Z:\gpu_heating\data.tif';

inputfile_psf = './psf.tif';%'Z:\gpu_heating\psf.tif';

g = gpuDevice;
reset(g);
wait(g);
g.FreeMemory;

resize_scale = 0.2;

dataA = double(ReadTifStack(path_data));
dataA = imresize(dataA, resize_scale);

N = size(dataA);

Estimate = gpuArray(dataA);
Measure = gpuArray(dataA);
 

FullPSF = double(ReadTifStack(inputfile_psf));
FullPSF = FullPSF/sum(FullPSF(:));
FullPSF = imresize(FullPSF, resize_scale);

OTF = fftn(circshift(gpuArray(FullPSF),-floor(N/2)));

iteration  = 200;
for k = 1: iteration
   
    t_start = clock;   
    Blur = double(real(ifftn(fftn(Estimate).*OTF))); 
    Diff = Measure./Blur;
    
    Correction = double(real(ifftn(fftn(Diff).*conj(OTF))));         
    
   
    Estimate = Estimate.* Correction; 
     
   
    disp(['iteration # ', num2str(k), ' takes ', num2str(etime(clock,t_start)), ' s']);
    
  
   if mod(k,40) == 0  % you can change iteration to any number to save intermediate results
    %   Estimate_Final = gather(gpu_Estimate);
        [pathname,filename,EXT] = fileparts(path_data);
        outputfile = strcat(pathname, '\', filename, '_iteration', num2str(k),'.tif');
       WriteTifStack(gather(Estimate), outputfile, 32);  % can save output as 16 bit, if the maximal value is less than 65535. 
   end

end
