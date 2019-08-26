% Requires Computer Vision ToolBox
% Requires Deep Learning Toolbox
% Requires Parallel Computing Toolbox
% Requires the Deep Learning Toolbox Importer for Keras Models support package. To install this support package, use the Add-On Explorer

%%
net = importKerasNetwork('matlab_mimicknet_small_ladj.h5', ...
                         'OutputLayerType', 'regression', ...
                         'ImageInputSize', [1808, 208])
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

% Plot Raw Beamformed
subplot(1,2,1);
imagesc(raw_iq);
colormap gray;
set(gca,'XColor', 'none','YColor','none')
title('Raw Beamformed');

% Plot MimickNet
subplot(1,2,2);
% 65 fps on Titan V for 300k pixels
% 6.2 fps on 4 core cpu for 300k pixels
% Note, this may take some time (60s) for the very first initialization.
for i = 1:4
    tic
    imagesc(predict(net, norm_iq, 'ExecutionEnvironment', 'gpu'));
    toc
end
colormap gray;
set(gca,'XColor', 'none','YColor','none')
title('MimickNet');
