function makeGrabCutResultFiles(datasetDir, resultDir, datasetSplitName)
%makeGrabCutResultFiles converts the results of the OpenCV GrabCut to the standard format

resultFolder = fullfile( resultDir, [datasetSplitName, '_CV']);
numFolds = 8;
cvFolder = fullfile( datasetDir, ['data_', datasetSplitName, '_CV']);
for iFold = 1 : numFolds
    fprintf('Working with fold %d / %d\n', iFold, numFolds);
    
    grabCutFileName = [datasetSplitName,'_fold', num2str(iFold, '%02d'), '_grabcut.mat'];
    
    resultFile = fullfile(resultFolder, grabCutFileName);
    
    load(fullfile(cvFolder, ['data_', datasetSplitName, '_fold', num2str(iFold, '%02d'),'_train.mat']), ...
        'objects');
    X_train = objects;
    load(fullfile(cvFolder, ['data_',datasetSplitName,'_fold', num2str(iFold, '%02d'),'_test.mat']), ...
        'objects');
    X_test = objects;
    
    Y_train_ans = cell(length(X_train), 1);
    Y_test_ans = cell(length(X_test), 1);
    Y_train = cell(length(X_train), 1);
    Y_test = cell(length(X_test), 1);
    
    for iObject = 1 : length(X_train)
        [~,imageName,~]=fileparts(X_train{iObject});
        Y_train{iObject} = readBinaryImage( fullfile( datasetDir, 'images-gt',[imageName,'.png'] ));
        Y_train_ans{iObject} = readBinaryImage( fullfile( resultDir, 'grabCutResults', [imageName,'.png'] ));
    end
    
    for iObject = 1 : length(X_test)
        [~,imageName,~]=fileparts(X_test{iObject});
        Y_test{iObject} = readBinaryImage( fullfile( datasetDir, 'images-gt', [imageName,'.png']) );
        Y_test_ans{iObject} = readBinaryImage(fullfile( resultDir, 'grabCutResults', [imageName,'.png'] ));
    end
    
    
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
        load(fullfile(datasetDir, X_train{iObject}), 'objSeed', 'bkgSeed', 'hammingWeights');
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
        load(fullfile(datasetDir, X_test{iObject}), 'objSeed', 'bkgSeed', 'hammingWeights');
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
    
end
