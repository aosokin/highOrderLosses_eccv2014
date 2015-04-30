function resultFile = runTraining(datasetDir, imageListDir, datasetName, resultFilesDir, loss, C, alpha)
%runTraining runs the training procedure of the SSVM

fprintf('Starting training.\n Dataset: %s\n Loss: %s\n', datasetName, loss);
%% load data
load(fullfile(imageListDir, ['data_', datasetName, '_train.mat']), 'objects', 'options');

%% prepare data
objectNumber = length(objects);
X_train = cell(objectNumber, 1);
Y_train = cell(objectNumber, 1);
for iObject = 1 : objectNumber
    X_train{iObject}.dataFile = fullfile(datasetDir, objects{iObject});

    load(fullfile(datasetDir, objects{iObject}), 'groundtruth');
    Y_train{iObject} = groundtruth;
end

%% training stage
% set the oracle and the loss
options.predictor = @oracle_graphCutSeeds_predictor;
options.oracle = [];
if strcmpi(loss, 'Hamming')
    options.oracle = @oracle_graphCutSeeds_hammingLoss_normalized;
end
if strcmpi(loss, 'hac')
    options.oracle = @oracle_graphCutSeeds_hacLoss_normalized;
end
if strcmpi(loss, 'area')
    options.oracle = @oracle_graphCutSeeds_areaLoss_normalized;
end
if strcmpi(loss, 'HammingWeighted')
    options.oracle = @oracle_graphCutSeeds_hammingWeightedLoss_normalized;
end
if strcmpi(loss, 'rowColumn')
    options.oracle = @oracle_graphCutSeeds_rowColumnLoss_normalized;
end
if strcmpi(loss, 'skeleton')
    options.oracle = @oracle_graphCutSeeds_skeletonLoss_normalized;
end
if isempty(options.oracle)
    error('runTraining:wrongLoss', 'Loss identifier is not recognized!')
end

% set the feature function
options.psi = @psi_images_binary;

% set the method parameters
options.eps = 0.001;
options.badConstraintEps = 1e-5;
options.maxIter = 100;

options.C = C;
options.distW = 1e+2;

if strcmpi(loss, 'skeleton')
    options.distWeightAlpha = alpha;
end


C = options.C;
options.QP_method = 'primal_matlab';
options.badConstraintWayout = 2;   % for reusing old training results
options.negativePairwiseWeights = true;
options.constrDropThreshhold = 500;
options.oraclePerQP = 1;
options.loadDataInMemory = true;

options.iterPerSave = inf;
tempFilesDir = fullfile( datasetDir, 'tempFiles' );
if ~isdir(tempFilesDir) && ~isinf(options.iterPerSave)
    mkdir(tempFilesDir);
end

if ~isdir(resultFilesDir)
    mkdir(resultFilesDir);
end
if strcmpi(loss, 'skeleton')
    options.tempFile = fullfile(tempFilesDir, [datasetName, '_', loss, '_C_', num2str(C), '_A_', num2str(alpha, '%10.0e'), '_lW_', num2str(options.distW) , '.mat']);
    resultFile = fullfile(resultFilesDir, [datasetName, '_', loss, '_C_', num2str(C), '_A_', num2str(alpha, '%10.0e'), '_lW_', num2str(options.distW),  '.mat']);
else
    options.tempFile = fullfile(tempFilesDir, [datasetName, '_', loss, '_C_', num2str(C), '_lossWeight_', num2str(options.distW) ,'.mat']);
    resultFile = fullfile(resultFilesDir, [datasetName, '_', loss, '_C_', num2str(C), '_lossWeight_', num2str(options.distW) ,'.mat']);
end

% start sSVM training
tStart = tic;
[W, Xi, trainingInfo] = train_sSVM_nSlack(X_train, Y_train, options);
time_train_sSVM_text_nSlack = toc(tStart);
fprintf('Training time: %f\n', time_train_sSVM_text_nSlack);

save(resultFile);

%% testing stage
% get test data
load(fullfile(imageListDir, ['data_', datasetName, '_test.mat']), 'objects');
objectNumber = length(objects);
X_test = cell(objectNumber, 1);
Y_test = cell(objectNumber, 1);
for iObject = 1 : objectNumber
    X_test{iObject}.dataFile = fullfile(datasetDir, objects{iObject});
    load(fullfile(datasetDir, objects{iObject}), 'groundtruth');
    Y_test{iObject} = groundtruth; 
end

% classify train set without object area
Y_train_ans = infer_images_binary(X_train, W, options);

% classify test set without object area
Y_test_ans = infer_images_binary(X_test, W, options);

%% compute the accuracies
% training losses
fprintf('Computing train losses\n');
trainLossHamming = nan(length(Y_train_ans), 1);
trainLossHammingWeighted = nan(length(Y_train_ans), 1);
trainLossArea = nan(length(Y_train_ans), 1);
trainLossRowColumn = nan(length(Y_train_ans), 1);
trainLossSkeleton = nan(length(Y_train_ans), 1);
for iObject = 1 : length(Y_train_ans)
    fprintf('*');
    curResult = Y_train_ans{iObject};
    curGt = Y_train{iObject};
    load(X_train{iObject}.dataFile, 'objSeed', 'bkgSeed', 'hammingWeights');
    curObjSeeds = objSeed;
    curBkgSeeds = bkgSeed;
    curHammingWeights = hammingWeights;
    
    trainLossHamming(iObject) = computeHammingLoss(curResult, curGt, curObjSeeds, curBkgSeeds);
    trainLossHammingWeighted(iObject) = computeHammingWeightedLoss(curResult, curGt, curHammingWeights, curObjSeeds, curBkgSeeds);
    trainLossArea(iObject) = computeAreaLoss(curResult, curGt, curObjSeeds, curBkgSeeds);
    trainLossRowColumn(iObject) = computeRowColumnLoss(curResult, curGt, curObjSeeds, curBkgSeeds);
    trainLossSkeleton(iObject) = computeSkeletonLoss(curResult, curGt, curObjSeeds, curBkgSeeds);
end
fprintf('\n');

% test losses
fprintf('Computing test losses\n');
testLossHamming = nan(length(Y_test_ans), 1);
testLossHammingWeighted = nan(length(Y_test_ans), 1);
testLossArea = nan(length(Y_test_ans), 1);
testLossRowColumn = nan(length(Y_test_ans), 1);
testLossSkeleton = nan(length(Y_test_ans), 1);
for iObject = 1 : length(Y_test_ans)
    fprintf('*');
    curResult = Y_test_ans{iObject};
    curGt = Y_test{iObject};
    load(X_test{iObject}.dataFile, 'objSeed', 'bkgSeed', 'hammingWeights');
    curObjSeeds = objSeed;
    curBkgSeeds = bkgSeed;
    curHammingWeights = hammingWeights;
    
    testLossHamming(iObject) = computeHammingLoss(curResult, curGt, curObjSeeds, curBkgSeeds);
    testLossHammingWeighted(iObject) = computeHammingWeightedLoss(curResult, curGt, curHammingWeights, curObjSeeds, curBkgSeeds);
    testLossArea(iObject) = computeAreaLoss(curResult, curGt, curObjSeeds, curBkgSeeds);
    testLossRowColumn(iObject) = computeRowColumnLoss(curResult, curGt, curObjSeeds, curBkgSeeds);
    testLossSkeleton(iObject) = computeSkeletonLoss(curResult, curGt, curObjSeeds, curBkgSeeds);
end
fprintf('\n');

save(resultFile);
   
% % plot results
% plot(1 : length(trainingInfo.fValuePlot2), trainingInfo.fValuePlot2, 'm', [1, length(trainingInfo.fValuePlot)], trainingInfo.fValue * ones(1, 2), 'k', 1 : length(trainingInfo.slackPlot), trainingInfo.slackPlot, 'r', 1 : length(trainingInfo.regPlot), trainingInfo.regPlot, 'g', 1 : length(trainingInfo.fValuePlot), trainingInfo.fValuePlot, 'b')
   
    
    
    
    
    
    
    
    


