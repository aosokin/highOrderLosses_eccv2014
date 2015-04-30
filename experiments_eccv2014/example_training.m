%example_training provides an example of how to train the SSVM model using this software
% see run_full_experiment.m for the full experiment of the ECCV 2014 paper 

rootDir = fileparts(fileparts(mfilename('fullpath')));
run(fullfile(rootDir, 'setup.m')) ;

datasetDir = fullfile(rootDir, 'data');
resultDir = fullfile(rootDir, 'resultFiles');

% precompute the fatures
% datasetName = prepareDataset( datasetDir );
datasetName = '8neighbors_63weights';

% prepare the splits
% datasetSplitName = prepareSplits_function_train_30_test_30( datasetDir, datasetName);
datasetSplitName = '8neighbors_63weights_train_30_test_30';

C = 1000;
alpha = 0.5;
imageListDir = fullfile( datasetDir, ['data_', datasetSplitName, '_CV']);
iFold = 2;
curFoldName = [datasetSplitName, '_', 'fold', num2str(iFold, '%02d')];
loss = 'Hamming';
resultFilesDir = fullfile( resultDir, [datasetSplitName, '_CV'] );

runTraining(datasetDir, imageListDir, curFoldName, resultFilesDir, loss, C, alpha);
