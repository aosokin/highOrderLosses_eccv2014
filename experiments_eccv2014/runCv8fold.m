function runCv8fold( datasetDir, datasetSplitName, resultDir )

experiment_C_set = { 100, 1000, 5000, 10000, 50000, 100000, 1000000 };
alpha = 0.5;

resultFilesDir = fullfile( resultDir, [datasetSplitName, '_CV'] );
if ~isdir(resultFilesDir)
    mkdir(resultFilesDir)
end

imageListDir = fullfile( datasetDir, ['data_', datasetSplitName, '_CV']);

numberOfFolds = 8;

for iExperiment = 1 : length(experiment_C_set)
    C = experiment_C_set{iExperiment};
    
    for iFold = 1 : numberOfFolds
        curFoldName = [datasetSplitName, '_', 'fold', num2str(iFold, '%02d')];
        
        loss = 'Hamming';
        runTraining(datasetDir, imageListDir, curFoldName, resultFilesDir, loss, C, alpha);
        
        loss = 'HammingWeighted';
        runTraining(datasetDir, imageListDir, curFoldName, resultFilesDir, loss, C, alpha);
        
        loss = 'area';
        runTraining(datasetDir, imageListDir, curFoldName, resultFilesDir, loss, C, alpha);
        
        loss = 'rowColumn';
        runTraining(datasetDir, imageListDir, curFoldName, resultFilesDir, loss, C, alpha);
        
        loss = 'skeleton';
        runTraining(datasetDir, imageListDir, curFoldName, resultFilesDir, loss, C, alpha);
    end
    
end

end

end
