function  datasetSplitName = prepareSplits_function_train_30_test_30(datasetDir, datasetName)
%prepareSplits_function_train_30_test_30 prepare splits used in the ECCV 2014 paper

%% fix the rand seed for the optimization
s = RandStream('mt19937ar', 'Seed', 3);
RandStream.setGlobalStream(s);

%% set dataset location
load(fullfile(datasetDir, ['features_', datasetName, '.mat']), 'dataset', 'options');

datasetSplitName = [datasetName, '_train_30_test_30'];

cvDir = fullfile(datasetDir, ['data_', datasetSplitName,  '_CV']);
if ~isdir(cvDir)
    mkdir(cvDir);
end

diffMask = findDifficultImages(dataset);
diffSet = dataset(diffMask);


%% make the splits
numberOfFolds = 8;
randomOrder = randperm(length(diffSet));
cvSet = randomOrder(1 : end);

foldSize = round(length(diffSet) / 2);

curTestSet = cell(numberOfFolds, 1);
curTrainSet = cell(numberOfFolds, 1);
for iFold = 1 : 2 : numberOfFolds
    randomOrder = randperm(length(diffSet));
    cvSet = randomOrder(1 : end);
    
    curTrainSet{iFold} = cvSet( 1 : foldSize);
    curTestSet{iFold} = setdiff( cvSet, curTrainSet{iFold}  );
    
    curTrainSet{iFold + 1} = cvSet( foldSize + 1 : end);
    curTestSet{iFold + 1} = setdiff( cvSet, curTrainSet{iFold + 1}  );
end

%% create and save splits of data
% create test for CV splits
for iFold = 1 : numberOfFolds
    
    foldSuffix = ['fold', num2str(iFold, '%02d')];
    
    curTestList = cell( 0, 1 );
    for iImage = 1 : length( curTestSet{iFold} )
        curTestList{ end + 1 } = diffSet{ curTestSet{iFold}( iImage ) };
    end
    objects = curTestList;
    save(fullfile(cvDir, ['data_', datasetSplitName, '_', foldSuffix ,'_test.mat']), 'objects', 'options');
    
    curTrainList = cell(0, 1 );
    for iImage = 1 : length( curTrainSet{iFold} )
        curTrainList{ end + 1 } = diffSet{ curTrainSet{iFold}( iImage ) };
    end
    objects = curTrainList;
    save(fullfile(cvDir, ['data_', datasetSplitName, '_', foldSuffix ,'_train.mat']), 'objects', 'options');
end

