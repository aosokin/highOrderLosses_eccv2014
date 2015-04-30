function makeLossTable( datasetName, resultDir )
%makeLossTable creates Table 2 of the ECCV 2014 paper

resultFolder = fullfile( resultDir, [datasetName, '_CV'] );
resultPrefix = datasetName;
resultSuffix = '.mat';

lossVals = cell(5, 1);
lossVals{1} = 'Hamming';
lossVals{2} = 'HammingWeighted';
lossVals{3} = 'area';
lossVals{4} = 'rowColumn';
lossVals{5} = 'skeleton';

cVals = [-1, 100, 1000, 5000, 10000, 50000, 100000, 1000000];
% '-1' corresponds to grabcut results

foldNumber = 8;

lossTableTrain_Hamming = nan( foldNumber, length(lossVals), length(cVals) );
lossTableTrain_HammingW = nan( foldNumber, length(lossVals), length(cVals) );
lossTableTrain_Area = nan( foldNumber, length(lossVals), length(cVals) );
lossTableTrain_rowColumn = nan( foldNumber, length(lossVals), length(cVals) );
lossTableTrain_skeleton = nan( foldNumber, length(lossVals), length(cVals) );

lossTableTest_Hamming = nan( foldNumber, length(lossVals), length(cVals) );
lossTableTest_HammingW = nan( foldNumber, length(lossVals), length(cVals) );
lossTableTest_Area = nan( foldNumber, length(lossVals), length(cVals) );
lossTableTest_rowColumn = nan( foldNumber, length(lossVals), length(cVals) );
lossTableTest_skeleton = nan( foldNumber, length(lossVals), length(cVals) );


for iFold = 1 : foldNumber
    for iLoss = 1 : length(lossVals)
        for iC = 1 : length(cVals)
            loss = lossVals{iLoss};
            C = cVals(iC);
            curName = ['_fold',num2str(iFold, '%02d'),'_',loss,'_C_', num2str(C)];
            if strcmpi(loss, 'skeleton')
                curName = [curName, '_A_5e-01_lW_100'];
            else
                curName = [curName, '_lossWeight_100'];
            end
            if C == -1
                curName = ['_fold',num2str(iFold, '%02d'), '_grabcut'];
            end
            
            fullName = fullfile(resultFolder, [ resultPrefix, curName, resultSuffix ]);
            
            if exist(fullName, 'file')
                load(fullName, 'trainLossHamming', 'trainLossHammingWeighted', 'trainLossArea', ...
                    'trainLossRowColumn', 'trainLossSkeleton', 'testLossHamming', ...
                    'testLossHammingWeighted', 'testLossArea', 'testLossRowColumn', ...
                    'testLossSkeleton');
                lossTableTrain_Hamming(iFold, iLoss, iC) = mean(trainLossHamming);
                lossTableTrain_HammingW(iFold, iLoss, iC) = mean(trainLossHammingWeighted);
                lossTableTrain_Area(iFold, iLoss, iC) = mean(trainLossArea);
                lossTableTrain_rowColumn(iFold, iLoss, iC) = mean(trainLossRowColumn);
                lossTableTrain_skeleton(iFold, iLoss, iC) = mean(trainLossSkeleton);
                
                lossTableTest_Hamming(iFold, iLoss, iC) = mean(testLossHamming);
                lossTableTest_HammingW(iFold, iLoss, iC) = mean(testLossHammingWeighted);
                lossTableTest_Area(iFold, iLoss, iC) = mean(testLossArea);
                lossTableTest_rowColumn(iFold, iLoss, iC) = mean(testLossRowColumn);
                lossTableTest_skeleton(iFold, iLoss, iC) = mean(testLossSkeleton);
            end
        end
    end
end

finalLoss = nan(6, 10);
cIds = 2 : 1 : length( cVals );
% hamming testing
[finalLoss(1, 2), bestCid_hamming_hamming] = min( mean(lossTableTest_Hamming(:, 1, cIds), 1), [], 3 );
finalLoss(1, 1) = mean(lossTableTrain_Hamming(:, 1, cIds(bestCid_hamming_hamming)), 1);

[finalLoss(2, 2), bestCid_hammingW_hamming] = min( mean(lossTableTest_Hamming(:, 2, cIds), 1), [], 3 );
finalLoss(2, 1) = mean(lossTableTrain_Hamming(:, 2, cIds(bestCid_hammingW_hamming)), 1);

[finalLoss(3, 2), bestCid_hammingArea_hamming] = min( mean(lossTableTest_Hamming(:, 3, cIds), 1), [], 3 );
finalLoss(3, 1) = mean(lossTableTrain_Hamming(:, 3, cIds(bestCid_hammingArea_hamming)), 1);

[finalLoss(4, 2), bestCid_hammingRC_hamming] = min( mean(lossTableTest_Hamming(:, 4, cIds), 1), [], 3 );
finalLoss(4, 1) = mean(lossTableTrain_Hamming(:, 4, cIds(bestCid_hammingRC_hamming)), 1);

[finalLoss(5, 2), bestCid_hammingSkel_hamming] = min( mean(lossTableTest_Hamming(:, 5, cIds), 1), [], 3 );
finalLoss(5, 1) = mean(lossTableTrain_Hamming(:, 5, cIds(bestCid_hammingSkel_hamming)), 1);

finalLoss(6, 1) = mean(lossTableTrain_Hamming(:, 1, 1), 1);
finalLoss(6, 2) = mean(lossTableTest_Hamming(:, 1, 1), 1);


% hamming weighted testing
[finalLoss(1, 4), bestCid_hamming_hammingW] = min( mean(lossTableTest_HammingW(:, 1, cIds), 1), [], 3 );
finalLoss(1, 3) = mean(lossTableTrain_HammingW(:, 1, cIds(bestCid_hamming_hammingW)), 1);

[finalLoss(2, 4), bestCid_hammingW_hammingW] = min( mean(lossTableTest_HammingW(:, 2, cIds), 1), [], 3 );
finalLoss(2, 3) = mean(lossTableTrain_HammingW(:, 2, cIds(bestCid_hammingW_hammingW)), 1);

[finalLoss(3, 4), bestCid_hammingArea_hammingW] = min( mean(lossTableTest_HammingW(:, 3, cIds), 1), [], 3 );
finalLoss(3, 3) = mean(lossTableTrain_HammingW(:, 3, cIds(bestCid_hammingArea_hammingW)), 1);

[finalLoss(4, 4), bestCid_hammingRC_hammingW] = min( mean(lossTableTest_HammingW(:, 4, cIds), 1), [], 3 );
finalLoss(4, 3) = mean(lossTableTrain_HammingW(:, 4, cIds(bestCid_hammingRC_hammingW)), 1);

[finalLoss(5, 4), bestCid_hammingSkel_hammingW] = min( mean(lossTableTest_HammingW(:, 5, cIds), 1), [], 3 );
finalLoss(5, 3) = mean(lossTableTrain_HammingW(:, 5, cIds(bestCid_hammingSkel_hammingW)), 1);

finalLoss(6, 3) = mean(lossTableTrain_HammingW(:, 1, 1), 1);
finalLoss(6, 4) = mean(lossTableTest_HammingW(:, 1, 1), 1);

% area testing
[finalLoss(1, 6), bestCid_hamming_Area] = min( mean(lossTableTest_Area(:, 1, cIds), 1), [], 3 );
finalLoss(1, 5) = mean(lossTableTrain_Area(:, 1, cIds(bestCid_hamming_Area)), 1);

[finalLoss(2, 6), bestCid_hammingW_Area] = min( mean(lossTableTest_Area(:, 2, cIds), 1), [], 3 );
finalLoss(2, 5) = mean(lossTableTrain_Area(:, 2, cIds(bestCid_hammingW_Area)), 1);

[finalLoss(3, 6), bestCid_hammingArea_Area] = min( mean(lossTableTest_Area(:, 3, cIds), 1), [], 3 );
finalLoss(3, 5) = mean(lossTableTrain_Area(:, 3, cIds(bestCid_hammingArea_Area)), 1);

[finalLoss(4, 6), bestCid_hammingRC_Area] = min( mean(lossTableTest_Area(:, 4, cIds), 1), [], 3 );
finalLoss(4, 5) = mean(lossTableTrain_Area(:, 4, cIds(bestCid_hammingRC_Area)), 1);

[finalLoss(5, 6), bestCid_hammingSkel_Area] = min( mean(lossTableTest_Area(:, 5, cIds), 1), [], 3 );
finalLoss(5, 5) = mean(lossTableTrain_Area(:, 5, cIds(bestCid_hammingSkel_Area)), 1);

finalLoss(6, 5) = mean(lossTableTrain_Area(:, 1, 1), 1);
finalLoss(6, 6) = mean(lossTableTest_Area(:, 1, 1), 1);


% row-column testing
[finalLoss(1, 8), bestCid_hamming_rowColumn] = min( mean(lossTableTest_rowColumn(:, 1, cIds), 1), [], 3 );
finalLoss(1, 7) = mean(lossTableTrain_rowColumn(:, 1, cIds(bestCid_hamming_rowColumn)), 1);

[finalLoss(2, 8), bestCid_hammingW_rowColumn] = min( mean(lossTableTest_rowColumn(:, 2, cIds), 1), [], 3 );
finalLoss(2, 7) = mean(lossTableTrain_rowColumn(:, 2, cIds(bestCid_hammingW_rowColumn)), 1);

[finalLoss(3, 8), bestCid_hammingArea_rowColumn] = min( mean(lossTableTest_rowColumn(:, 3, cIds), 1), [], 3 );
finalLoss(3, 7) = mean(lossTableTrain_rowColumn(:, 3, cIds(bestCid_hammingArea_rowColumn)), 1);

[finalLoss(4, 8), bestCid_hammingRC_rowColumn] = min( mean(lossTableTest_rowColumn(:, 4, cIds), 1), [], 3 );
finalLoss(4, 7) = mean(lossTableTrain_rowColumn(:, 4, cIds(bestCid_hammingRC_rowColumn)), 1);

[finalLoss(5, 8), bestCid_hammingSkel_rowColumn] = min( mean(lossTableTest_rowColumn(:, 5, cIds), 1), [], 3 );
finalLoss(5, 7) = mean(lossTableTrain_rowColumn(:, 5, cIds(bestCid_hammingSkel_rowColumn)), 1);

finalLoss(6, 7) = mean(lossTableTrain_rowColumn(:, 1, 1), 1);
finalLoss(6, 8) = mean(lossTableTest_rowColumn(:, 1, 1), 1);

% skeleton testing
[finalLoss(1, 10), bestCid_hamming_skeleton] = min( mean(lossTableTest_skeleton(:, 1, cIds), 1), [], 3 );
finalLoss(1, 9) = mean(lossTableTrain_skeleton(:, 1, cIds(bestCid_hamming_skeleton)), 1);

[finalLoss(2, 10), bestCid_hammingW_skeleton] = min( mean(lossTableTest_skeleton(:, 2, cIds), 1), [], 3 );
finalLoss(2, 9) = mean(lossTableTrain_skeleton(:, 2, cIds(bestCid_hammingW_skeleton)), 1);

[finalLoss(3, 10), bestCid_hammingArea_skeleton] = min( mean(lossTableTest_skeleton(:, 3, cIds), 1), [], 3 );
finalLoss(3, 9) = mean(lossTableTrain_skeleton(:, 3, cIds(bestCid_hammingArea_skeleton)), 1);

[finalLoss(4, 10), bestCid_hammingRC_skeleton] = min( mean(lossTableTest_skeleton(:, 4, cIds), 1), [], 3 );
finalLoss(4, 9) = mean(lossTableTrain_skeleton(:, 4, cIds(bestCid_hammingRC_skeleton)), 1);

[finalLoss(5, 10), bestCid_hammingSkel_skeleton] = min( mean(lossTableTest_skeleton(:, 5, cIds), 1), [], 3 );
finalLoss(5, 9) = mean(lossTableTrain_skeleton(:, 5, cIds(bestCid_hammingSkel_skeleton)), 1);

finalLoss(6, 9) = mean(lossTableTrain_skeleton(:, 1, 1), 1);
finalLoss(6, 10) = mean(lossTableTest_skeleton(:, 1, 1), 1);

disp(finalLoss);








