% The BioMassters Challenge, MATLAB Solution
% Team: D_R_K_A (Azin Al Kajbaf, Kaveh Faraji)
% The competition benchmark code has been used to develop this solution.
% Therefore, parts of the code could be similar to the benchmark code. 
% Please refer to "The BioMassters Challenge Starter Code” by Grace Woolson.
% Please run the "bm_conv1d.m" file before running the "main.m".
%% Providing data path  
s3Path = 'data/';
%% Importing and Pre-processing the Data
agbmFolder = fullfile(s3Path, 'train_agbm');
%% Min and max of 15 channels in images, scale images between 0 & 1
channel_min = [-22.9017, -39.6357, -25.0000, -66.4466,...
               0.0000,   0.0000,   0.0000,...
          0.0000,   0.0000,   0.0000,   0.0000, ...
          0.0000,   0.0000,   0.0000, 0.0000, 0.00];

channel_max = [27.5460, 21.3467, 26.2766, 21.0031,...
               12134.9092, 12430.0000,...
        12928.1816, 11060.4170, 10803.3633, 10791.7275, 13006.8184, 10734.3330,...
         9111.6670,  9190.5000,   255.0, 650.0];
mat_min = zeros(16, 12 , 256, 256);
mat_max = zeros(16, 12 , 256, 256);
for i=1:16
    mat_min (i,:,:,:) = channel_min(i);
    mat_max (i,:,:,:) = channel_max(i);
end

%% Input data
imInput = imageDatastore(agbmFolder, 'ReadFcn', @(filename)readTrainingSatelliteData(filename, s3Path,mat_min, mat_max), 'LabelSource', 'foldernames');
[inputTrain,inputVal] = splitEachLabel(imInput,0.9,0.1);
%% Output data
imOutput = imageDatastore(agbmFolder, 'ReadFcn', @(filename)readlabel(filename, s3Path), 'LabelSource', 'foldernames');
[outputTrain,outputVal] = splitEachLabel(imOutput,0.9,0.1);

%% Combining input and label data
dsTrain = combine(inputTrain, outputTrain);
dsVal = combine(inputVal, outputVal);

%% Building a 3-D Unet network
lgraph = unet3dLayers([12 256 256 16], 2,'encoderDepth',2, 'NumFirstEncoderFilters',16);
input_layer = image3dInputLayer([12 256 256 16],"Name","input","Normalization","zscore");
lgraph = lgraph.replaceLayer('ImageInputLayer', input_layer);
lgraph = lgraph.removeLayers('Softmax-Layer');
lgraph = lgraph.removeLayers('Segmentation-Layer');
finalConvolutionLayer = convolution3dLayer([1, 1, 1], 1, 'Name', 'Final-ConvolutionLayer-3D');
lgraph = lgraph.replaceLayer('Final-ConvolutionLayer', finalConvolutionLayer);
lgraph = lgraph.addLayers(averagePooling3dLayer([12,1,1], "Name","avg-poll-3D" ));
lgraph = lgraph.addLayers(regressionLayer('name','regressionLayer'));
lgraph = lgraph.connectLayers('Final-ConvolutionLayer-3D','avg-poll-3D');
lgraph_1 = lgraph.connectLayers('avg-poll-3D','regressionLayer');
analyzeNetwork(lgraph_1);

%% Setting training preferences
options = trainingOptions('adam', ...
    'LearnRateSchedule','piecewise', ...
    'MiniBatchSize', 4, ...
    'MaxEpochs', 20, ...
    'InitialLearnRate',0.001, ...
    'LearnRateDropFactor',0.3,...
    'LearnRateDropPeriod',10,...
    'ValidationData', dsVal, ...
    'OutputNetwork', 'best-validation-loss', ...
    'Verbose', true,...
    'ValidationFrequency', 1000,...
    'Plots','training-progress', ...
    'OutputNetwork', 'best-validation-loss');  % ExecutionEnvironment="multi-gpu",    'OutputNetwork', 'best-validation-loss', ...
%% Training the model
train_network = false; %If you want to train network from start 
if train_network
    net = trainNetwork(dsTrain, lgraph_1, options);
    save('trainedNetwork.mat','net')
else
    % load pretrained
    load('pretrained/trainedNetwork_v8.mat');
    lgraph_1 = layerGraph(net);
    analyzeNetwork(lgraph_1);
end

%% Calculating model error on validation data
% Predicting Validation labels
reset(dsVal)
for ii=1:size(dsVal.UnderlyingDatastores{1,1}.Files,1)
    input_label = dsVal.read; % reading input and label
    predict_test{ii} = predict(net,input_label{1},'ExecutionEnvironment','gpu'); % Predicting labels
    testBatch{ii} = input_label{2};
end
for idx = 1:size(predict_test, 2)
    predicted = predict_test{idx};
    ref = testBatch{idx};
    rmse_mat{idx} = sqrt(mse(ref, predicted));
end
% Error of predicting validation data (rmse)
mean([rmse_mat{:}])
%% Reading testing metadata
featuresMetadataLocation = fullfile(s3Path, 'features_metadata.csv');
featuresMetadata = readtable(featuresMetadataLocation, 'ReadVariableNames',true);
testFeatures = featuresMetadata(strcmp(featuresMetadata.split, 'test'), :);
testChips = testFeatures.chip_id;
[~, uniqueIdx, ~] = unique(testChips);
uniqueTestChips = testChips(uniqueIdx, :);

%% Directory for saving test data predictions
if ~exist('test_agbm', 'dir')
    mkdir test_agbm
end
outputFolder = './test_agbm/';

%% Predicting test data
for chipIDNum = 1:length(uniqueTestChips)
    chip_id = uniqueTestChips{chipIDNum};

    % Format inputs
    inputImage = readTestingSatelliteData(chip_id, s3Path, mat_min, mat_max);
 
    % Make predictions
    pred = squeeze(predict(net, inputImage));
 
    % Set up TIF file and export predictions
    filename = [outputFolder, chip_id, '_agbm.tif'];
    t = Tiff(filename, 'w');
 
    % Need to set tag info of Tiff file
    tagstruct.ImageLength = 256; 
    tagstruct.ImageWidth = 256;
    tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
    tagstruct.BitsPerSample = 32;
    tagstruct.SamplesPerPixel = 1;
    tagstruct.SampleFormat = Tiff.SampleFormat.IEEEFP;
    tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky; 
    tagstruct.Compression = Tiff.Compression.None;
    tagstruct.Software = 'MATLAB'; 
    setTag(t,tagstruct)
    % Export your prediction
    write(t, pred);
    close(t);
end

%% Helper functions

% This helper function reads the input data and prepares them for training the network.
% The helpers function deals with missing with replacing them with the
% nearest available channel or month (dependent on what is missing in satellite images).
function ImS1S2_scaled = readTrainingSatelliteData(outputFilename, s3Path, mat_min, mat_max)
outputFilenameParts = split(outputFilename, ["_", "/"]);
chip_id = outputFilenameParts{end-1};
inputDir = fullfile(s3Path, 'train_features/');
correspondingFiles = dir([inputDir, chip_id, '*.tif']);
s1Data = cell(1, 12);
s2Data = cell(1, 12);
S1Data_mat = zeros(12,256,256,4);
S2Data_mat = zeros(12,256,256,11);
% Compiling and ordering all data
for fileIdx = 1:length(correspondingFiles)
    filename = correspondingFiles(fileIdx).name;
    filenameParts = split(filename, ["_", "/", "."]);
    satellite = filenameParts{end-2};
    fullfilename = strcat(inputDir, filename);
    im = imread(fullfilename);
    idx = str2double(filenameParts{end-1}) + 1;

    if satellite == 'S1'
        s1Data{idx} = fullfilename;
        S1Data_mat(idx,:,:,:)=im;
    elseif satellite == 'S2'
        s2Data{idx} = fullfilename;
        S2Data_mat(idx,:,:,:)=im;
    end
end

channels_avg = [ -11.4333,  -18.0622,  -11.3462,  -17.9976, 1661.5421, 1648.6449,1638.1118,...
    1960.0146, 2524.0505, 2633.5132, 2780.6536, 2727.5730, 1039.4055,  707.1373,   21.0693];

% Dealing with the missing data, S2
not_empty_cells = find(~cellfun(@isempty,s2Data));
for imgNum = 1:12
    if isempty(s2Data{imgNum})
        [m2, I2] = min(abs(imgNum-not_empty_cells));
        S2Data_mat(imgNum,:,:,:)=S2Data_mat(not_empty_cells(I2),:,:,:);
    end
end

% Dealing with the missing data, S1
chk_chan = zeros(1, 4, 12);
for imgNum = 1:12
    if ismember(-9999, S1Data_mat(imgNum,:,:,:))
        for chnNum = 1:4
            if ismember(-9999, S1Data_mat(imgNum,:,:,chnNum))
                chk_chan(1, chnNum, imgNum)=1;
            end
        end

    end
end

for imgNum = 1:12

    for chnNum = 1:4

        if chk_chan(1, chnNum, imgNum)==1

            if any(chk_chan(1,chnNum,:)==0)
                not_empty_chan = find(chk_chan(1,chnNum,:)==0);
                [m1, I1] = min(abs(imgNum - not_empty_chan));
                S1Data_mat(imgNum,:,:,chnNum)=S1Data_mat(not_empty_chan(I1),:,:,chnNum);
            else

                S1Data_mat(imgNum,:,:,chnNum)=channels_avg(1,chnNum);
            end

        end
    end

end

% Reading labels generated by 1-D CNN step (the "bm_conv1d.m" file)
predDir = fullfile('predictions_1D/');
correspondingFiles_pred_conv1d = dir([predDir, chip_id, '_agbm.tif']);
pred_im = imread(strcat('predictions_1D/', correspondingFiles_pred_conv1d.name));
pred_im_add_dim(1,:,:) = pred_im;

pred_im_add_dim = repmat( pred_im_add_dim , 12,1,1);

% Combining all bands into one 16-band image
ImS1S2 = cat(4, S1Data_mat, S2Data_mat, pred_im_add_dim);
ImS1S2 = single(permute(ImS1S2,[4,1,2,3]));
% Scaling images between 0 & 1
ImS1S2_scaled = (ImS1S2- mat_min) ./ (mat_max - mat_min);
ImS1S2_scaled = single(permute(ImS1S2_scaled,[2, 3, 4, 1]));

end


% Reading testing data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ImS1S2_scaled = readTestingSatelliteData(chip_id, s3Path, mat_min, mat_max)
inputDir = fullfile(s3Path, 'test_features/');
correspondingFiles = dir([inputDir, chip_id, '*.tif']);
% Preallocating a cell arrrays & matrices
s1Data = cell(1, 12);
s2Data = cell(1, 12);
S1Data_mat = zeros(12,256,256,4);
S2Data_mat = zeros(12,256,256,11);
% Compiling and order all data
for fileIdx = 1:length(correspondingFiles)
    filename = correspondingFiles(fileIdx).name;
    filenameParts = split(filename, ["_", "/", "."]);
    satellite = filenameParts{end-2};
    fullfilename = strcat(inputDir, filename);
    im = imread(fullfilename);
    idx = str2double(filenameParts{end-1}) + 1;
    if satellite == 'S1'
        s1Data{idx} = fullfilename;
        S1Data_mat(idx,:,:,:)=im;
    elseif satellite == 'S2'
        s2Data{idx} = fullfilename;
        S2Data_mat(idx,:,:,:)=im;
    end
end

channels_avg = [ -11.4333,  -18.0622,  -11.3462,  -17.9976, 1661.5421, 1648.6449,1638.1118,...
    1960.0146, 2524.0505, 2633.5132, 2780.6536, 2727.5730, 1039.4055,  707.1373,   21.0693]; 

% Dealing with the missing data, S2
not_empty_cells = find(~cellfun(@isempty,s2Data));
for imgNum = 1:12
    if isempty(s2Data{imgNum})
        [m2, I2] = min(abs(imgNum-not_empty_cells));
        S2Data_mat(imgNum,:,:,:)=S2Data_mat(not_empty_cells(I2),:,:,:);
    end
end

% Dealing with the missing data, S1
chk_chan = zeros(1, 4, 12);
for imgNum = 1:12
    if ismember(-9999, S1Data_mat(imgNum,:,:,:))
        for chnNum = 1:4
            if ismember(-9999, S1Data_mat(imgNum,:,:,chnNum))
                chk_chan(1, chnNum, imgNum)=1;
            end
        end

    end
end

for imgNum = 1:12

    for chnNum = 1:4

        if chk_chan(1, chnNum, imgNum)==1

            if any(chk_chan(1,chnNum,:)==0)
                not_empty_chan = find(chk_chan(1,chnNum,:)==0);
                [m1, I1] = min(abs(imgNum - not_empty_chan));
                S1Data_mat(imgNum,:,:,chnNum)=S1Data_mat(not_empty_chan(I1),:,:,chnNum);
            else
                S1Data_mat(imgNum,:,:,chnNum)=channels_avg(1,chnNum);
            end

        end
    end

end

% Reading labels generated by 1-D CNN step
predDir = fullfile('predictions_1D/');
correspondingFiles_pred_conv1d = dir([predDir, chip_id, '_agbm.tif']);
pred_im = imread(strcat('predictions_1D/', correspondingFiles_pred_conv1d.name));
pred_im_add_dim(1,:,:) = pred_im;
pred_im_add_dim = repmat( pred_im_add_dim , 12,1,1);

% Combining all bands into one 16-band image
ImS1S2 = cat(4, S1Data_mat, S2Data_mat, pred_im_add_dim);
ImS1S2 = single(permute(ImS1S2,[4,1,2,3]));

ImS1S2_scaled = (ImS1S2- mat_min) ./ (mat_max - mat_min);
ImS1S2_scaled = single(permute(ImS1S2_scaled,[2, 3, 4, 1]));
end


function label_output = readlabel(outputFilename, s3Path)
outputFilenameParts = split(outputFilename, ["_", "/"]);
chip_id = outputFilenameParts{end-1};
inputDir = fullfile(s3Path, 'train_agbm/');
correspondingFiles = dir([inputDir, chip_id, '*.tif']);
filename = correspondingFiles.name;
fullfilename = strcat(inputDir, filename);
im = imread(fullfilename);
x(1,:,:) = im;
label_output= x;
end
