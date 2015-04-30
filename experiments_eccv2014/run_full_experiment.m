%run_full_experiment reproduces Table 2 of our ECCV 2014 paper

rootDir = fileparts(fileparts(mfilename('fullpath')));
run(fullfile(rootDir, 'setup.m')) ;

datasetDir = fullfile(rootDir, 'data');
resultDir = fullfile(rootDir, 'resultFiles');

% precompute the fatures
datasetName = prepareDataset( datasetDir );
% datasetName = '8neighbors_63weights';

% prepare the splits
datasetSplitName = prepareSplits_function_train_30_test_30( datasetDir, datasetName);
% datasetSplitName = '8neighbors_63weights_train_30_test_30';

% run training for all the folds of the cross-validation
runCv8fold( datasetDir, datasetSplitName, resultDir )

% run tht OpenCV GrabCut code
system(['python runGrabCutOpenCv.py ', fullfile(datasetDir, 'images'), ' ', fullfile(datasetDir, 'images-labels-large'), ' ', fullfile(resultDir, 'grabCutResults') ]);

% convert the results of the GrabCut to the stadard format
makeGrabCutResultFiles(datasetDir, resultDir, datasetSplitName);

% produce Table 2
makeLossTable( datasetSplitName, resultDir )
