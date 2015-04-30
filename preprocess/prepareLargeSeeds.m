function prepareLargeSeeds( imageDir, initialSeedsDir, groundtruthDir, seedsDir )
% prepareLargeSeeds enlarges the human-provided seeds to make the segmentation task easier
%
% Anton Osokin (firstname.lastname@gmail.com)


% get the image list
imageFilesDir = dir(fullfile(imageDir));
imageNumber = 0;
imageFiles = cell(imageNumber, 1);
imageNames = cell(imageNumber , 1);
for iImage = 1 : length(imageFilesDir)
    if (length(imageFilesDir(iImage).name) > 4) % if the current name looks like the valid name
        imageNumber = imageNumber + 1;
        
        imageFiles{imageNumber} = imageFilesDir(iImage).name;
        [~, imageNames{imageNumber}] = fileparts(imageFiles{imageNumber});
    end
end
clear('imageFilesDir', 'iImage');

% set up the target folder
dilationLevel = 0.5;
targetDir = seedsDir;
if ~isdir(targetDir)
    mkdir(targetDir);
end
   
for iImage = 1 : imageNumber
    curSeed = imread(fullfile(initialSeedsDir, [imageNames{iImage}, '-anno.png']));
    curGroundtruth = im2double(imread(fullfile(groundtruthDir, [imageNames{iImage}, '.png'])));
    
    objSeedMask = curSeed == 1;
    bkgSeedMask = curSeed == 2;
    
    objSeedDistTransform = bwdist(objSeedMask);
    bkgSeedDistTransform = bwdist(bkgSeedMask);
    
    objectBoundary = bwperim(curGroundtruth > 0.1);
    distBoundary = bwdist(objectBoundary);
    
    innerDist = distBoundary;
    innerDist(~ (curGroundtruth > 0.9)) = 0;
    
    objSeedBoundaryDist = mean(objSeedDistTransform(objectBoundary));
    newObjSeedMask = objSeedDistTransform <= dilationLevel * objSeedBoundaryDist;
    newObjSeedMask = (newObjSeedMask & (innerDist > 5)) | innerDist > 20 | objSeedMask;
    
    outerDist = distBoundary;
    outerDist( curGroundtruth > 0.1) = 0;
  
    bkgSeedBoundaryDist = mean(bkgSeedDistTransform(objectBoundary));
    newBkgSeedMask = bkgSeedDistTransform <= dilationLevel * bkgSeedBoundaryDist;
    
    newBkgSeedMask = (newBkgSeedMask & (outerDist > 5)) | outerDist > 20 | bkgSeedMask;
    
    newSeed = uint8(zeros(size(curSeed)));
    newSeed(newObjSeedMask) = 1;
    newSeed(newBkgSeedMask) = 2;
        
    imwrite(newSeed, [0 0 0; 1 1 1; 1 0 0], fullfile(targetDir, [imageNames{iImage}, '-anno.png']), 'png');
end
    
end
