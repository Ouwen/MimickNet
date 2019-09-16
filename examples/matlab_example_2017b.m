% Requires Computer Vision ToolBox
% Requires Deep Learning Toolbox
% Requires Parallel Computing Toolbox

%%
layers = importKerasLayers('matlab_mimicknet_small_ladj.h5', ...
                         'OutputLayerType', 'regression', ...
                         'ImportWeights', true)
layers = removeLayers(layers,'input_1');
layers = addLayers(layers,imageInputLayer([1808, 208, 1], 'Normalization', 'none', 'Name', 'input_1'));
layers = connectLayers(layers,'input_1','conv2d_1');

XTrain = ones([1808, 208, 1,2]);
YTrain = XTrain;

options = trainingOptions('sgdm', 'MaxEpochs',1, 'InitialLearnRate',eps('double'), 'MiniBatchSize',2);
net = trainNetwork(XTrain,YTrain,layers,options);

%%
figure;
load('mark_liver.mat');
norm_iq = abs(iq); % Real data only

% Clip to -80, normalize to [0,1], and pad to be divisible by 16
norm_iq = 20*log10(norm_iq/max(norm_iq(:))); 
norm_iq(norm_iq <-80) = -80;
norm_iq = (norm_iq - min(norm_iq(:)))/(max(norm_iq(:)) - min(norm_iq(:)));
norm_iq = make_shape(norm_iq, 16);

raw_iq = abs(iq);
raw_iq = 20*log10(raw_iq/max(raw_iq(:))); 
raw_iq(raw_iq <-50) = -50;
raw_iq = (raw_iq - min(raw_iq(:)))/(max(raw_iq(:)) - min(raw_iq(:)));
raw_iq = make_shape(raw_iq, 16);

% Plot Delay and Sum
subplot(1,2,1);
imagesc(raw_iq);
colormap gray;
set(gca,'XColor', 'none','YColor','none')
title('Delay and Sum');

% Plot MimickNet
subplot(1,2,2);
% 65 fps on Titan V for 300k pixels
% 6.2 fps on 4 core cpu for 300k pixels
% Note, this may take some time (60s) for the very first initialization.

tic
imagesc(predict(net, norm_iq, 'ExecutionEnvironment', 'cpu')); % set to gpu for compute 3.0 gpus
toc
colormap gray;
set(gca,'XColor', 'none','YColor','none')
title('MimickNet');
