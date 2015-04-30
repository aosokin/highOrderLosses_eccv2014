%example_motivation provides a motivating example from ECCV 2014 paper

rootDir = fileparts(fileparts(mfilename('fullpath')));
run(fullfile(rootDir, 'setup.m')) ;

gtImages = cell(2,1);
gtImages{1} = readBinaryImage_noGray('groundTruth/006_insectjaw.png');
gtImages{2} = readBinaryImage_noGray('groundTruth/010_blackbeetle.png');

resultImages = cell(2, 3);
resultImages{1, 1} = readBinaryImage_noGray('segmentations/006_insectjaw_GT_open.png');
resultImages{1, 2} = readBinaryImage_noGray('segmentations/006_insectjaw_GT_average.png');
resultImages{1, 3} = readBinaryImage_noGray('segmentations/006_insectjaw_GT_thinning.png');
resultImages{2, 1} = readBinaryImage_noGray('segmentations/010_blackbeetle_GT_open.png');
resultImages{2, 2} = readBinaryImage_noGray('segmentations/010_blackbeetle_GT_average.png');
resultImages{2, 3} = readBinaryImage_noGray('segmentations/010_blackbeetle_GT_thinning.png');

lossTable = nan( 4, 6 );
for iImage = 1 : 2
    for iResult = 1 : 3
        curResult = resultImages{iImage, iResult};
        lossTable(1 , (iImage - 1) * 3  + iResult) = computeHammingLoss( curResult, gtImages{iImage} );
        lossTable(2 , (iImage - 1) * 3  + iResult) = computeJaccardLoss( curResult, gtImages{iImage} );
        lossTable(3 , (iImage - 1) * 3  + iResult) = computeAreaLoss( curResult, gtImages{iImage} );
        lossTable(4 , (iImage - 1) * 3  + iResult) = computeSkeletonLoss( curResult, gtImages{iImage} );
    end
end
    
lossTable = lossTable * 100;

disp( lossTable )
