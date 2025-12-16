clc; clear;
% Set folders
imageFolder = '/home/user/archana/synthetic_dataset/images'; % your folder with input images
imageFiles = dir(fullfile(imageFolder, '*.jpg'));
% Initialize accumulators
totalPSNR = 0;
totalSSIM = 0;
numImages = length(imageFiles);
% --- Define Network (same as before) ---
layers = [
  imageInputLayer([128 128 2],'Name','input')
  convolution2dLayer(3,16,'Padding','same','Name','conv1')
  reluLayer('Name','relu1')
  maxPooling2dLayer(2,'Stride',2,'Name','pool1')
  convolution2dLayer(3,8,'Padding','same','Name','conv2')
  reluLayer('Name','relu2')
  maxPooling2dLayer(2,'Stride',2,'Name','pool2')
  transposedConv2dLayer(3,8,'Stride',2,'Cropping','same','Name','upconv1')
  reluLayer('Name','relu3')
  transposedConv2dLayer(3,16,'Stride',2,'Cropping','same','Name','upconv2')
  reluLayer('Name','relu4')
  convolution2dLayer(1,1,'Padding','same','Name','finalconv')
  regressionLayer('Name','output')
];
% --- Training Data Accumulation ---
XTrain = [];
YTrain = [];
inputImages = {};   % To store input watermarked images
outputImages = {};  % To store processed/cleaned images
for i = 1:numImages
  % Load and convert to Lab
  imgPath = fullfile(imageFolder, imageFiles(i).name);
  inputImage = im2double(imread(imgPath));
  labImage = rgb2lab(inputImage);
  L = labImage(:,:,1) / 100;
  a = labImage(:,:,2);
  b = labImage(:,:,3);
  % Edge detection on L
  edges = imgradient(L, 'sobel');
  edges = mat2gray(edges);
  % Otsu-based WM detection
  wm_mask = imbinarize(edges, graythresh(edges));
  wm_mask = imresize(wm_mask, [128 128]);
  % Prepare inputs
  inputStack = cat(3, L, edges);
  inputStack = imresize(inputStack, [128 128]);
  targetL = imresize(L, [128 128]);
  XTrain(:,:,:,i) = inputStack;
  YTrain(:,:,:,i) = targetL;
end
% --- Train Network ---
options = trainingOptions('adam', ...
  'InitialLearnRate',1e-3, ...
  'MaxEpochs',20, ...
  'MiniBatchSize',1, ...
  'Shuffle','every-epoch', ...
  'Verbose',false, ...
  'Plots','training-progress');
net = trainNetwork(XTrain, YTrain, layers, options);
% --- Test Each Image and Compute PSNR/SSIM ---
for i = 1:numImages
  imgPath = fullfile(imageFolder, imageFiles(i).name);
  inputImage = im2double(imread(imgPath));
  labImage = rgb2lab(inputImage);
  L = labImage(:,:,1) / 100;
  a = labImage(:,:,2);
  b = labImage(:,:,3);
  % Edge detection and WM detection
  edges = imgradient(L, 'sobel');
  edges = mat2gray(edges);
  wm_mask = imbinarize(edges, graythresh(edges));
  wm_mask = imresize(wm_mask, [128 128]);
  % Prepare input
  inputStack = cat(3, L, edges);
  inputStack = imresize(inputStack, [128 128]);
  % Predict
  predictedL = predict(net, reshape(inputStack, [128 128 2 1]));
  predictedL = squeeze(predictedL);
  predictedL = imresize(predictedL, size(L));
  % Resize mask
  wm_mask_full = imresize(wm_mask, size(L));
  % Replace only watermark areas
  outputL = L;
  outputL(wm_mask_full > 0) = predictedL(wm_mask_full > 0);
  outputL = outputL * 100;
  % Reconstruct and convert to RGB
  reconstructedLab = cat(3, outputL, a, b);
  reconstructedRGB = lab2rgb(reconstructedLab);
  % Convert to uint8 for metrics
  inputImage8 = im2uint8(inputImage);
  reconstructed8 = im2uint8(reconstructedRGB);
  % Compute PSNR and SSIM
  psnrVal = psnr(reconstructed8, inputImage8);
  ssimVal = ssim(reconstructed8, inputImage8);
  % fprintf('Image %d: PSNR = %.2f dB, SSIM = %.4f\n', i, psnrVal, ssimVal);
  totalPSNR = totalPSNR + psnrVal;
  totalSSIM = totalSSIM + ssimVal;
  inputImages{end+1} = inputImage;
  outputImages{end+1} = reconstructedRGB;
   brisqueVal = brisque(reconstructed8);
  niqeVal    = niqe(reconstructed8);

  % Optional: if you want masked evaluation (on watermark region only)
  % Here, we crop patches overlapping with wm_mask_full
  maskedB = []; maskedN = [];
  tile = 96; step = 48;
  for yy = 1:step:size(L,1)-tile+1
    for xx = 1:step:size(L,2)-tile+1
        patchM = wm_mask_full(yy:yy+tile-1, xx:xx+tile-1);
        if mean(patchM(:)) > 0.5   % mostly watermark area
            patchI = reconstructed8(yy:yy+tile-1, xx:xx+tile-1, :);
            maskedB(end+1) = brisque(patchI);
            maskedN(end+1) = niqe(patchI);    
        end
    end
  end
  if ~isempty(maskedB)
      brisqueMasked = mean(maskedB);
      niqeMasked    = mean(maskedN);
  else
      brisqueMasked = NaN;
      niqeMasked    = NaN;
  end

  % Accumulate
  totalBRISQUE(i) = brisqueVal;
  totalNIQE(i)    = niqeVal;
  maskedBRISQUE(i)= brisqueMasked;
  maskedNIQE(i)   = niqeMasked;
end
figure;
for k = 1:min(2, length(outputImages))
   subplot(2,2,2*k-1); imshow(inputImages{k}); title(sprintf('Watermarked Image %d', k));
   subplot(2,2,2*k);   imshow(outputImages{k}); title(sprintf('Denoised Output %d', k));
end
% --- Final Averages ---
avgPSNR = totalPSNR / numImages;
avgSSIM = totalSSIM / numImages;
fprintf('\nAverage PSNR: %.2f dB\n', avgPSNR);
fprintf('Average SSIM: %.4f\n', avgSSIM);
% save('preprocNet.mat','net');  % 'net' is your trained network
% fprintf('Preprocessing DNN saved as preprocNet.mat\n');
  % Compute BRISQUE and NIQE (no-reference metrics)
 
  avgBRISQUE = mean(totalBRISQUE);
  avgNIQE    = mean(totalNIQE);
  avgMaskedBRISQUE = mean(maskedBRISQUE,'omitnan');
  avgMaskedNIQE    = mean(maskedNIQE,'omitnan');
  fprintf('Average BRISQUE: %.2f\n', avgBRISQUE);
  fprintf('Average NIQE   : %.2f\n', avgNIQE);
  fprintf('Average Masked BRISQUE: %.2f\n', avgMaskedBRISQUE);
  fprintf('Average Masked NIQE   : %.2f\n', avgMaskedNIQE);

