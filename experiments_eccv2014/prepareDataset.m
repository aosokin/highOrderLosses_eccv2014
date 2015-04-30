function datasetName = prepareDataset( datasetDir )
%prepareDataset prepares the dataset used in the ECCV 2014 paper

%% fix the rand seed for the optimization
s = RandStream('mt19937ar', 'Seed', 3);
RandStream.setGlobalStream(s);

%% read the data
% set up the directories
imageDir = fullfile(datasetDir, 'images');
groundtruthDir = fullfile(datasetDir, 'images-gt');
seedsDir = fullfile(datasetDir, 'images-labels-large');
initialSeedsDir = fullfile(datasetDir, 'images-labels');

% get the image list
imageFiles = dir(fullfile(imageDir));
numImages = 0;
imageFile = cell(numImages, 1);
for iImage = 1 : length(imageFiles)
    if (length(imageFiles(iImage).name) > 4) % if the current name looks like the valid name
        numImages = numImages + 1;
        imageFile{numImages} = imageFiles(iImage).name;
    end
end

% prepare the increased seeds
fprintf('Preparing seeds ... ');
prepareLargeSeeds( imageDir, initialSeedsDir, groundtruthDir, seedsDir );
fprintf('done\n');

%% initialize the data arrays
% initialize options
options = struct;
options.classNum = 2;

options_preprocess = struct;
options_preprocess.connectivity = 'colourGradient8';
options_preprocess.different_edge_type = 1;
options_preprocess.numberOfGaussians = (5);

edgeGroups{1} = [1, 3];
edgeGroups{2} = [2, 4];
edgeTypeNumber = 2;
    
options_preprocess.seedDir = seedsDir;

pixelData = cell(numImages, 1);

datasetName = '8neighbors_63weights';
dataFolderName = ['features_', datasetName];
dataDir = fullfile(datasetDir, dataFolderName);
if ~isdir(dataDir)
    mkdir(dataDir);
end

options.unaryFeatureNum = 51;
options.pairwiseFeatureNum = 12;

dataset = cell(numImages, 1);

% precomputed coefficients to normalize the maximum of features to be 1
 unaryFeatureMax = [];
 unaryFeatureMin = [];

%% prepare all the features
for iImage = 1 : numImages
    fprintf('Working with image %d of %d\n', iImage, numImages );
    [~, imageName_cur] = fileparts(imageFile{iImage});
    
    % Run hol code
    curObject = preprocessImage(imageFile{iImage}, [datasetDir, '/'], options_preprocess);
    image_cur = curObject.image;
    unaryFeatures_cur = curObject.features_unary;
    nodeMap_cur = reshape(1 : size(curObject.features_unary, 2), [size(image_cur, 1), size(image_cur, 2)]);
    objSeed_cur = curObject.label_seed == 1;
    bkgSeed_cur = curObject.label_seed == 2;
    groundtruth_cur = curObject.label;
    
    % extractLuv features
    luvFeatures = extractLuvGmmUnaryFeatures( curObject.image, curObject.label_seed, options_preprocess.numberOfGaussians );

    % compute the distance transform features
    distUnaryFeatures = getDistanceFeatures( image_cur, curObject.label_seed );
    
    % combine all unary features
    unaryFeatures_cur = [unaryFeatures_cur; luvFeatures'; distUnaryFeatures'];
   
    % normalize unary features
    if isempty(unaryFeatureMax) || isempty(unaryFeatureMin)
        unaryFeatureMin = min(unaryFeatures_cur, [], 2);
        unaryFeatureMax = max(unaryFeatures_cur, [], 2);
    end
    
    mask = ( unaryFeatureMax ~= unaryFeatureMin );
    unaryFeatures_cur(mask, :) =  bsxfun(@rdivide, unaryFeatures_cur(mask, :), (unaryFeatureMax(mask) - unaryFeatureMin(mask)));
    
    % create pairwise feature vector
    curEdgeColorDiffs = (curObject.edgeColorWeights);
    autoBeta = 1 / ( 0.5 * mean(curEdgeColorDiffs));
    oldWeights = exp( -autoBeta * curEdgeColorDiffs );
    newWeights = [exp( -autoBeta * 10 * curEdgeColorDiffs ), exp( -autoBeta * 3 * curEdgeColorDiffs ), exp( -autoBeta * 0.3 * curEdgeColorDiffs ), exp( -autoBeta * 0.1 * curEdgeColorDiffs )];
    allPairwiseWeights = [oldWeights, newWeights, ones( size(curEdgeColorDiffs) )];
    
    
    % combine classes of pairwise edges
    pairwiseFeatures_cur = [curObject.edges', nan(size(curObject.edges, 2), size(allPairwiseWeights, 2) * edgeTypeNumber)];
    minId = min(curObject.edges_type);
    maxId = max(curObject.edges_type);

    
    for iType = 1 : edgeTypeNumber
        iPos = (iType - 1) * size(allPairwiseWeights, 2) + 3;
        curWeights = zeros(size(curObject.edges, 2), size(allPairwiseWeights, 2));
        
        mask = false(size(curObject.edges_type ));
        for iDir = 1 : length(edgeGroups{iType})
            mask = mask | (curObject.edges_type == minId + edgeGroups{iType}(iDir) - 1);
        end
        
        curWeights(mask, :) = allPairwiseWeights(mask, :);
        pairwiseFeatures_cur(:, iPos : iPos + size(allPairwiseWeights, 2) - 1) = curWeights;
    end
    
    % save data to files
    imageName = imageName_cur;
    dataset{iImage} = fullfile(dataFolderName, [imageName, '.mat']);
    unaryFeatures = single(unaryFeatures_cur);
    pairwiseFeatures = single(pairwiseFeatures_cur);
    nodeMap = int32(nodeMap_cur);
    objSeed = objSeed_cur;
    bkgSeed = bkgSeed_cur;
    groundtruth = groundtruth_cur;
    image = single(image_cur);

    % compute the Hamming weight from the groundtruth
    curMask = double(groundtruth == 1 | isnan(groundtruth));
    boundaryDistance = bwdist(bwperim(curMask));
    hammingWeights = 1 + 10 * exp( - boundaryDistance / 7);
    normalization = numel(hammingWeights) / sum(hammingWeights(:));
    hammingWeights = single(hammingWeights * normalization);

    save(fullfile(datasetDir, dataset{iImage}), 'image', 'imageName', 'unaryFeatures', 'pairwiseFeatures', 'nodeMap', 'objSeed', 'bkgSeed', 'groundtruth', 'hammingWeights');
end

%% save the dataset file
save(fullfile(datasetDir, ['features_', datasetName, '.mat']), 'dataset', 'options');

end
