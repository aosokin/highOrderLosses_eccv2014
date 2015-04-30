function makeUserStudySlides
%makeUserStudySlides creates figures for userStudy.tex

% setup
rootDir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
run(fullfile(rootDir, 'setup.m'));
datasetDir = fullfile(rootDir, 'data' );
imageDir = fullfile(datasetDir, 'images' );
groundTruthDir = fullfile(datasetDir, 'images-gt' );
resultsDir = fullfile( fileparts( mfilename('fullpath') ), 'results' );

lossName = { 'hamming', 'area', 'rowColumn', 'skeleton' };
userStudyImages = readFileList( 'userStudyImages.txt' );

%% fix the rand seed for the optimization
s = RandStream('mt19937ar', 'Seed', 3);
RandStream.setGlobalStream(s);

%% prepare Latex Table
outputDir = fullfile( 'figures' );
if ~isdir(outputDir)
    mkdir(outputDir);
end

tableFile = fopen( fullfile( outputDir, 'userStudyImages.tex'), 'w');
objPrintedCounter = 0;
for iObject = 1 : length( userStudyImages )
    matchingFiles = dir( fullfile(imageDir, [userStudyImages{iObject}, '.*']) );
    imageFile = fullfile(imageDir, matchingFiles(1).name);
    copyfile(imageFile, fullfile(outputDir, matchingFiles(1).name) );
    curImage = imread( imageFile );
    
    curFileNames = cell( length(lossName), 1 );
    curLossOrder = randperm( length(lossName) );
    
    for iLossOrder = 1 : length(lossName)
        iLoss = curLossOrder( iLossOrder );

        curMask = imread( fullfile( resultsDir, lossName{iLoss},  [userStudyImages{iObject},'-res.png'] ) );
        maskedImage = createMaskedImage( curImage, curMask, [1 0 0] );
        
        imageName = fullfile( outputDir, [ userStudyImages{iObject}, '_', num2str(iLoss), '.jpg'] );
        imwrite(maskedImage, imageName);
        
        curFileNames{iLossOrder} = imageName;
    end
    
    gtFile = fullfile( groundTruthDir, [userStudyImages{iObject}, '.png'] );
    copyfile(gtFile, fullfile( outputDir, [userStudyImages{iObject}, '-gt.png']) );

    widthStr = '3.7';
    if size(curImage, 1) > 1.2 * size(curImage, 2)
        widthStr = '2.5';
    end
    
    objPrintedCounter = objPrintedCounter + 1;
    fprintf(tableFile, '\\begin{frame} \n');
    fprintf(tableFile, '\\frametitle{Image %d} \n', objPrintedCounter);
    fprintf(tableFile, '\\begin{center} \n');
    fprintf(tableFile, '\\begin{tabular}{c@{\\qquad}c@{}c} \n');
    fprintf(tableFile, '\\includegraphics[width=%scm]{figures/%s} & \n', widthStr,  matchingFiles(1).name);
    fprintf(tableFile, '\\includegraphics[width=%scm]{%s} & \n', widthStr, curFileNames{1});
    fprintf(tableFile, '\\includegraphics[width=%scm]{%s} \\\\[-0.1cm] \n', widthStr, curFileNames{2});
    fprintf(tableFile, '\\includegraphics[width=%scm]{figures/%s} & \n', widthStr,  [userStudyImages{iObject}, '-gt.png']);
    fprintf(tableFile, '\\includegraphics[width=%scm]{%s} & \n', widthStr, curFileNames{3});
    fprintf(tableFile, '\\includegraphics[width=%scm]{%s} \n', widthStr, curFileNames{4});
    fprintf(tableFile, '\\end{tabular} \n');
    fprintf(tableFile, '\\end{center} \n');
    fprintf(tableFile, '\\end{frame} \n\n');
    
end
fclose(tableFile);

end

function files = readFileList( fileListName )

fileID = fopen(fileListName, 'r');
if fileID == -1
    error(['File ', fileListName, ' can not be opened!']);
end
files = textscan(fileID, '%s\n');
fclose(fileID);

files = files{1};

end

function result = createMaskedImage(img, masks, colors)

numLabels = size(masks, 3);
if (size(img, 3) == 1)
    res1 = im2double(img);
    res2 = res1;
    res3 = res1;
else
    imgD = im2double(img);
    res1 = imgD(:, :, 1);
    res2 = imgD(:, :, 2);
    res3 = imgD(:, :, 3);
end

for k = 1 : numLabels
    mask = masks(:, :, k);
    color = colors(k, :);
    
    mask = bwmorph(mask,'open');
    mask = mask & ~bwmorph(mask,'erode', 3);
    
    res1(mask) = color(1);
    res2(mask) = color(2);
    res3(mask) = color(3);
end

result = cat(3, res1, res2, res3);

end
