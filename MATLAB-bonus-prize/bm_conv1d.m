% The BioMassters Challenge, MATLAB Solution
% Team: D_R_K_A (Azin Al Kajbaf, Kaveh Faraji)
% The competition benchmark code has been used to develop this solution.
% Therefore, parts of the code could be similar to the benchmark code. 
% Please refer to "The BioMassters Challenge Starter Code” by Grace Woolson.
%% Providing data path
s3Path = 'data/';
%% Importing and Pre-processing the Data
agbmFolder = fullfile(s3Path, 'train_agbm');
%% Min and max of 15 channels in images, scale images between 0 & 1
channel_min = [-22.9017, -39.6357, -25.0000, -66.4466,...
               0.0000,   0.0000,   0.0000,...
          0.0000,   0.0000,   0.0000,   0.0000, ...
          0.0000,   0.0000,   0.0000, 0.0000];

channel_max = [27.5460, 21.3467, 26.2766, 21.0031,...
               12134.9092, 12430.0000,...
        12928.1816, 11060.4170, 10803.3633, 10791.7275, 13006.8184, 10734.3330,...
         9111.6670,  9190.5000,   255.0];
mat_min = zeros(15, 12 , 256, 256);
mat_max = zeros(15, 12 , 256, 256);
for i=1:15
    mat_min (i,:,:,:) = channel_min(i);
    mat_max (i,:,:,:) = channel_max(i);
end
%% Input data
imInput = imageDatastore(agbmFolder, 'ReadFcn', @(filename)readTrainingSatelliteData(filename, s3Path, mat_min, mat_max), 'LabelSource', 'foldernames');
[inputTrain,inputVal] = splitEachLabel(imInput,0.9,0.1);

%% Output data
imOutput = imageDatastore(agbmFolder, 'ReadFcn', @(filename)readlabel(filename, s3Path), 'LabelSource', 'foldernames');
[outputTrain,outputVal] = splitEachLabel(imOutput,0.9,0.1);

%% Combining inputs and labels for training
dsTrain = combine(inputTrain, outputTrain);
dsVal = combine(inputVal, outputVal);
%% Buidling a 1-D CNN
filterSize = 3;
numFilters = 64;
numClasses=10;
layers = [ ...
    sequenceInputLayer(15, MinLength=12)
    convolution1dLayer(filterSize,numFilters,Padding="causal")
    reluLayer
    layerNormalizationLayer
    maxPooling1dLayer(2)
    convolution1dLayer(filterSize,2*numFilters,Padding="causal")
    reluLayer
    layerNormalizationLayer
    globalMaxPooling1dLayer
    fullyConnectedLayer(512)
    reluLayer
    fullyConnectedLayer(128)
    reluLayer
    fullyConnectedLayer(1)
    reluLayer];
% analyzeNetwork(layers);
lgraph = layerGraph(layers);
net = dlnetwork(lgraph);

%% Defining minibatchqueue to load data for training loops
mbq = minibatchqueue(dsTrain,...
    MiniBatchSize=1,...
    MiniBatchFcn=@preprocessMiniBatch);%MiniBatchFormat=["" ""]
mbq_val = minibatchqueue(dsVal,...
    MiniBatchSize=1,...
    MiniBatchFcn=@preprocessMiniBatch);%MiniBatchFormat=["" ""]
% [X, T]= next(mbq); % check data

%% Using the pre-trained model or training the model

% If you want to train the model from scratch, change "train_network" value
% to true.
train_network = true; 

if train_network
    miniBatchSize = 1;
    numEpochs = 20;
    numObservations = numel(inputTrain.Files);
    numIterationsPerEpoch = floor(numObservations./miniBatchSize);
    averageGrad = [];
    averageSqGrad = [];
    numIterations = numEpochs * numIterationsPerEpoch;
    monitor = trainingProgressMonitor(Metrics="Loss",Info="Epoch",XLabel="Iteration");

    iteration = 0;
    epoch = 0;

    while epoch < numEpochs && ~monitor.Stop
        epoch = epoch + 1;

        % Shuffle data.
        shuffle(mbq);
        reset(mbq_val);
        while hasdata(mbq) && ~monitor.Stop

            iteration = iteration + 1;

            % Read mini-batch of data.
            [X,T] = next(mbq);
            % Convert mini-batch of data to a dlarray.
            X = dlarray(single(X),"CBT");
            % We read each image with the size of [15 * 12 * 256 * 256] and 
            % convert it to a [channel_size(C) = 15 batch_size(B) = (256*256) temporal_size(T) = 12]

            % We had to use a batch size smaller than 65501.
            % We got errors when we used batch size above this value.

            X = X(:,1:65500,:);
            T = T(:,1:65500,:);
            % If training on a GPU, then convert data to a gpuArray.
            if canUseGPU
                X = gpuArray(X);
                T= gpuArray(T);
            end

            % Calculate loss and gradients using the helper loss function.
            [loss,gradients] = dlfeval(@modelLoss,net,X,T);
            % Update the network parameters using the Adam optimizer.
            [net,averageGrad,averageSqGrad] = adamupdate(net,gradients,averageGrad,averageSqGrad,iteration);
            % Update the training progress monitor.
            recordMetrics(monitor,iteration,Loss=loss);
            updateInfo(monitor,Epoch=epoch + " of " + numEpochs);
            monitor.Progress = 100 * iteration/numIterations;

        end

        % Validation error
        ii=0;
        while hasdata(mbq_val)
            [X_val, T_val]= next(mbq_val);
            if canUseGPU
                X_val = gpuArray(dlarray(single(X_val),"CBT"));
                T_val = gpuArray(T_val);
            end
            Y_val = predict(net,X_val);
            error = mse(Y_val, T_val)^.5;
            ii=ii+1;
            rmse_error(ii) = extractdata(gather(error));
%           disp(rmse_error(ii))
        end
        disp(['Epoch'+ string(epoch)+' Validation Error(RMSE): ' , mean(rmse_error)])

    end
    save('trainedNetwork_conv1d.mat','net')
else
    lgraph = load('pretrained/trainedNetwork_conv1d.mat');% Load pre-trained network
    net = lgraph.net;

end

%% Calculating RMSE of validation data

ii=0;
reset(mbq_val)
while hasdata(mbq_val)
    [X_val, T_val]= next(mbq_val);
    if canUseGPU
        X_val = gpuArray(dlarray(single(X_val),"CBT"));
        T_val = gpuArray(T_val);
    end
    Y_val = predict(net,X_val);
    error = mse(Y_val, T_val)^.5;
    ii=ii+1;
    rmse_error(ii) = extractdata(gather(error));
end
disp(mean(rmse_error))

%% Reading training and testing data features

featuresMetadataLocation = fullfile(s3Path, 'features_metadata.csv');
featuresMetadata = readtable(featuresMetadataLocation, 'ReadVariableNames',true);
testFeatures = featuresMetadata(strcmp(featuresMetadata.split, 'test'), :);
trainFeatures = featuresMetadata(strcmp(featuresMetadata.split, 'train'), :);
testChips = testFeatures.chip_id;
trainChips = trainFeatures.chip_id;
[~, uniqueIdx_train, ~] = unique(trainChips);
[~, uniqueIdx_test, ~] = unique(testChips);
uniqueTestChips = testChips(uniqueIdx_test, :);
uniqueTrainChips = trainChips(uniqueIdx_train, :);

%% Creating a folder to save predictions of the 1-D CNN

if ~exist('predictions_1D', 'dir')
    mkdir predictions_1D
end
outputFolder = './predictions_1D/';

%% Predicting the labels using 1-D CNN

% If you want to produce data for unet network the "train_data" value should be true.
% Change it to false if you just want to predict testing data
train_data = true; 

if train_data
uniqueChips =  {uniqueTrainChips{:} uniqueTestChips{:}};
else
    uniqueChips =  uniqueTestChips;
end

for chipIDNum = 1:length(uniqueChips)
    
    chip_id = uniqueChips{chipIDNum};
 
    % Format inputs
    if (chipIDNum <= length(uniqueTrainChips)) &&  (train_data)
        inputImage = readTestingSatelliteData(chip_id, s3Path, mat_min, mat_max, true);% reading train data
    else
        inputImage = readTestingSatelliteData(chip_id, s3Path, mat_min, mat_max, false);% reading test data
    end
    X_test = gpuArray(dlarray(single(inputImage),"CBT"));
    % Make predictions
    pred_temp = predict(net, X_test);
    pred = reshape(pred_temp,[256, 256]);
    pred = extractdata(gather(pred));
    % Set up TIF file and export prediction
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
    % Add all input images to ordered cell array
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

% Combining all bands into one 15-band image
ImS1S2 = cat(4, S1Data_mat, S2Data_mat);
ImS1S2 = single(permute(ImS1S2,[4,1,2,3]));


% Scaling images between 0 and 1
ImS1S2_scaled = (ImS1S2- mat_min) ./ (mat_max - mat_min);
ImS1S2_scaled = reshape(ImS1S2_scaled,[15,12,256*256]);
% Changing axis 
ImS1S2_scaled = single(permute(ImS1S2_scaled,[1, 3, 2]));

end


% This helper function reads the input data and prepares them for prediction step.
% The helpers function deals with missing with replacing them with the
% nearest available channel or month (dependent on what is missing in satellite images).
function ImS1S2_scaled = readTestingSatelliteData(chip_id, s3Path, mat_min, mat_max, train_data)
% Location of input data
if train_data
    inputDir = fullfile(s3Path, 'train_features/');
else
    inputDir = fullfile(s3Path, 'test_features/');
end
correspondingFiles = dir([inputDir, chip_id, '*.tif']);

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
% Combining all bands into one 15-band image
ImS1S2 = cat(4, S1Data_mat, S2Data_mat);
ImS1S2 = single(permute(ImS1S2,[4,1,2,3]));

ImS1S2_scaled = (ImS1S2- mat_min) ./ (mat_max - mat_min);
ImS1S2_scaled = reshape(ImS1S2_scaled,[15,12,256*256]);
ImS1S2_scaled = single(permute(ImS1S2_scaled,[1, 3, 2]));
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
x = reshape(x,[1,256*256]);
label_output= x;
end


function [X,T] = preprocessMiniBatch(dataX,dataT)

% Pre-process predictors
X = dataX{:};
% Label
T = dataT{:}; 

end

function [loss,gradients] = modelLoss(net,X,T)

% Forward data through network
Y = forward(net,X);

% MSE loss
loss = mse(Y,T);
% loss = (mse(Y,T)+1e6)^.5; #RMSE loss

% Calculating gradients of loss with respect to learnable parameters.
gradients = dlgradient(dlarray(loss),net.Learnables);

end
