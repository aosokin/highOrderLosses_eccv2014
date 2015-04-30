function [F, C, warmStart] = oracle_graphCutSeeds_skeletonLoss_normalized(iObject, Y, W, config)
%oracle_graphCutSeeds_skeletonLoss_normalized - oracle for SSVM: loss augmented MAP-inference;
% binary segmentation, pairwise MRF, Potts pairwise weights, graphcut, user-defined seeds
% Silhouette loss normalized to have maximum of 1
% Separation oracle that finds maximum violated constraint
% W' * Psi(X, Y) >= L(W) - Xi
% L(W) is a linear function on W: L(W) = W' * F + C
% In undergenerating approach L(W) = W' * Psi(X, Y_worst) + loss(Y, Y_worst)
% Y_worst = argmax_(\bar{Y}) ( W' * Psi(X, \bar{Y}) + loss(\bar{Y}, Y) )
% object data is supposed to be stored in global variable X_dataset
%
% [F, C, warmStart] = oracle_graphCutSeeds_skeletonLoss_normalized(iObject, Y, W, config);
%
% INPUT
%   iObject - index of object X, integer 1 x 1'
%   Y - labeling Y; double imageHeight x imageWidth;
%   W - current weight vector W; numWeights x 1;
%   config - structure od method parameters:
%       K - number of classes
%       unaryFeatureNum - number fo unary features
%       pairwiseFeatureNum - number of pairwise features
%       loadDataInMemory - flag for loading all data in memory
%       distW - weight for the loss function       
%       
% OUTPUT
%   F - coefficient; double numWeights x 1;
%   C - constant term in linear constraint; double 1 x 1;
%   warmStart - warm start for the next oracle on the same object (if required);
%
%   Anton Osokin, 27.11.2012

%% initialization
% fprintf('Starting GraphCut oracle: ');
tStart = tic;

% This oracle does not have a warmStart options
warmStart = [];

% Number of classes
K = config.K;

% number of nodes
N = length(Y(:));

% Extract unary features
unaryWeights = W(1 : config.unaryFeatureNum);
% Extract pairwise features
pairwiseWeights = W(config.unaryFeatureNum + 1 : config.unaryFeatureNum + config.pairwiseFeatureNum);

%% extract data related to the current object
% load data
requiredVariablesList = {'unaryFeatures', 'pairwiseFeatures', 'nodeMap', 'objSeed', 'bkgSeed'};
[variables, variablesLoaded] = loadVariableGlobalDataset(requiredVariablesList, iObject, config.loadDataInMemory);
if any(~variablesLoaded)
    badVarStr = strjoin(requiredVariablesList(~variablesLoaded), '; ');
    error([mfilename,':dataNotLoaded'], ['Could not load ', badVarStr, ' for object #', num2str(iObject)]);
end
unaryFeatures = double(variables{1});
pairwiseFeatures = double(variables{2});
nodeMap = double(variables{3});
nodeNum = max(nodeMap(:));
objSeed = variables{4};
bkgSeed = variables{5};

%% construct sets for the loss
xPos = repmat( 1 : size(Y, 2), [size(Y, 1), 1] );
yPos = repmat( (1 : size(Y, 1))', [1, size(Y, 2)] );

% construct sets from skeleton to the boundary
boundary = bwmorph(Y, 'remove');
skeleton = bwmorph(Y, 'skel', inf);
skeletonPoints = find(skeleton(:));
distToBoundary = bwdist(boundary);

% decrease the number of skeleton points
skeletonPoints = skeletonPoints(1 : 4 : end);
radiusWeights = [1.25, 1.25 * 0.5, 1.25 * 0.25];
constrSetSize = length(skeletonPoints) * length(radiusWeights);
setNodeIndex = cell(constrSetSize, 1);
setNodeWeight = cell(constrSetSize, 1);
toDelete = true(constrSetSize, 1);
notNanY = ~isnan(Y);
iSet = 0;
for iPoint = 1 : length(skeletonPoints)
    xCenter = xPos(skeletonPoints(iPoint));
    yCenter = yPos(skeletonPoints(iPoint));
    curDist = distToBoundary(skeletonPoints(iPoint));
    
    for iRadius = 1 : length(radiusWeights)
        
        radius = curDist * radiusWeights(iRadius);
        mask = (xPos - xCenter).^2 + (yPos - yCenter).^2 < radius ^ 2;
        mask = mask & notNanY;
    
        iSet = iSet + 1;
        setNodeIndex{iSet} = find(mask(:));
    
        if numel(setNodeIndex{iSet}) < 1
            toDelete(iSet) = true;
        else
            toDelete(iSet) = false;
        end
    
        setNodeWeight{iSet} = ones(length(setNodeIndex{iSet}), 1);
    end
end

setNodeIndex(toDelete) = [];
setNodeWeight(toDelete) = [];
constrSetSize = length(setNodeIndex);

setWeight = nan(constrSetSize, 1);
setSum = nan(constrSetSize, 1);
for iSet = 1 : constrSetSize
    setSum(iSet) = sum(Y(setNodeIndex{iSet}) .* setNodeWeight{iSet} );
    setWeight(iSet) = 1 / numel(setNodeIndex{iSet});
end

%% set the weight for the loss
[variables, flagAdded] = loadVariableGlobalDataset({'normalizationFactorSkeleton', 'normalizationFactorHamming'}, iObject, config.loadDataInMemory);
if all(flagAdded)
    normalizationFactorSkeleton = variables{1};
    normalizationFactorHamming = variables{2};
else
    maxSkeletonLoss = getMaxSetHammingLoss( Y, 1.0, setNodeIndex, setNodeWeight, setSum, setWeight, objSeed, bkgSeed);
    curSetWeight = setWeight / maxSkeletonLoss;
    maxFullLoss = getMaxSetHammingLoss( Y, config.distWeightAlpha, setNodeIndex, setNodeWeight, setSum, curSetWeight, objSeed, bkgSeed);
    
    maxHammingLoss = sum( objSeed(:) == 0 & bkgSeed(:) == 0 & ~isnan(Y(:)));
    
    normalizationFactorSkeleton = maxSkeletonLoss / config.distWeightAlpha * maxFullLoss;
    normalizationFactorHamming = maxHammingLoss / (1 - config.distWeightAlpha) * maxFullLoss;
    
    addVariableGlobalDataset('normalizationFactorSkeleton', iObject, normalizationFactorSkeleton);
    addVariableGlobalDataset('normalizationFactorHamming', iObject, normalizationFactorHamming);
end

lossSkeletonWeight = config.distW / normalizationFactorSkeleton;
lossHammingWeight = config.distW / normalizationFactorHamming;

% add Hamming weight to the loss
lossWeights = lossWeights_Hamming(Y(:));
lossWeights = lossWeights * lossHammingWeight;


%% construct the energy
% Multiply features by weights
uF = bsxfun(@times, unaryFeatures, unaryWeights);
pF = bsxfun(@times, pairwiseFeatures(:, 3 : end), pairwiseWeights');

% Compute data costs
dataCost = [zeros(nodeNum, 1), sum(uF, 1)'];

dataCost(:, 1) = dataCost(:, 1) + accumarray(nodeMap(:), lossWeights(1, :)', [nodeNum 1], @sum, 0);
dataCost(:, 2) = dataCost(:, 2) + accumarray(nodeMap(:), lossWeights(2, :)', [nodeNum 1], @sum, 0);

% incorporate seeds
seedWeight = max(max(abs(dataCost(:)) * 1000), 1000);
dataCost(objSeed(:) == 1, 1) = -seedWeight;
dataCost(bkgSeed(:) == 1, 2) = -seedWeight;

%% Build a graph
termWeights = double([-dataCost(:, 2), -dataCost(:, 1)]);
tmp = sum(pF, 2);
tmp(tmp > 0) = 0;
edgeWeights = [pairwiseFeatures(:, 1), pairwiseFeatures(:, 2), -tmp, -tmp];

%% augment energy with a loss function
% add new unary nodes and their unary potentials
zIndexSet = nan(constrSetSize, 1);
newTermWeights = nan(1000000, 2);
newTermWeights(1 : size(termWeights, 1), :) = termWeights;
endIndex = size(termWeights, 1);
for iSet = 1 : constrSetSize
    tmpNode = [(2 * lossSkeletonWeight) * setSum(iSet) * setWeight(iSet), 0];  % add an extra node
    newTermWeights(endIndex + 1 : endIndex + size(tmpNode, 1), :) = tmpNode;
    endIndex = endIndex + size(tmpNode, 1);
    
    zIndexSet(iSet) = endIndex; % index of the new node
    newTermWeights(setNodeIndex{iSet}, 1) = newTermWeights(setNodeIndex{iSet}, 1) - setNodeWeight{iSet} * lossSkeletonWeight * setWeight(iSet); % modify the unaries
end
newTermWeights(any(isnan(newTermWeights), 2), :) = [];
termWeights = newTermWeights;

%add edges related to the new node
newEdgeWeights = nan(15000000, 4);
newEdgeWeights(1 : size(edgeWeights, 1), :) = edgeWeights;
endIndex = size(edgeWeights, 1);
for iSet = 1 : constrSetSize
    curOnes = ones(length(setNodeIndex{iSet}), 1);
    tmpEdges = [zIndexSet(iSet) * curOnes, setNodeIndex{iSet}, (2 * lossSkeletonWeight) * setNodeWeight{iSet} * setWeight(iSet), 0 * curOnes];
    newEdgeWeights(endIndex + 1 : endIndex + size(tmpEdges, 1), :) = tmpEdges;
    endIndex = endIndex + size(tmpEdges, 1);
end
newEdgeWeights(any(isnan(newEdgeWeights), 2), :) = [];
edgeWeights = newEdgeWeights;

%% Run GraphCut
try
    [energy_worst, nodeLabels_worst] = graphCutMex(termWeights, edgeWeights);
catch err
    save('error_graphCut.mat', 'termWeights', 'edgeWeights');
    error('sSVM:oracle: error in graph cut');
end
zValue = nodeLabels_worst(zIndexSet);
nodeLabels_worst(zIndexSet) = [];
Y_worst = nodeLabels_worst(nodeMap);

%% Analyze the results
% check if seed constraints are satisfied
if any(Y_worst(:) .* bkgSeed(:) > 0) || any((1 - Y_worst(:)) .* objSeed(:) > 0)
    warning('ERROR IN ORACLE: seeds did not work!');
end

% Compute linear bound
% compute unary features
uF = sum(unaryFeatures(:, nodeLabels_worst == 1), 2);
% compute pairwise features
activePotts = (nodeLabels_worst(pairwiseFeatures(:, 1)) ~= nodeLabels_worst(pairwiseFeatures(:, 2)));
pF = sum(pairwiseFeatures(activePotts, 3 : end), 1)';
F = [uF; pF];

% F2 = psi_images_binary(iObject, Y_worst, config);
% if any(F ~= F2)
%     error('ERROR IN ORACLE: computing constraints!');
% end

% compute the constant loss-based term for a loss
C = 0;
for iSet = 1 : constrSetSize
    C = C + lossSkeletonWeight * setWeight(iSet) * (2 * zValue(iSet) * ( sum(Y_worst(setNodeIndex{iSet}) .* setNodeWeight{iSet}) - setSum(iSet) ) - sum(Y_worst(setNodeIndex{iSet}) .* setNodeWeight{iSet}) + setSum(iSet));
end

% add Hamming weights
C = C + sum(lossWeights(Y_worst(:) + 1 + 2 * (0 : numel(Y_worst) - 1)'));


% check if computed linear bound equals the energy computed by GraphCut
checkDiff = abs((-energy_worst + lossSkeletonWeight * sum(setSum .* setWeight)) - (F' * W + C));
if  checkDiff > 1e-4
    warning(['ERROR IN ORACLE: computing the loss-augmented energy, error size: ', num2str(checkDiff)]);
end
    

tOracle = toc(tStart);
% fprintf('oracle time: %f\n', tOracle);
end

function lossWeights = lossWeights_Hamming(Y2)
% Hamming loss between Y1 and Y2
% loss should be equal to sum_i sum_k [Y1(i) = k] * lossWeights(k, i)
% Y2 is a vector that contains 0, 1, NaN
% for NaN positions loss should be always zero

% Get parameters
K = 2;
N = size(Y2, 1);

Y2_new = Y2 + 1;
Y2_new(isnan(Y2)) = K + 1;

% Compute lossWeights
lossWeights = [1 - eye(K), zeros(K, 1)];

lossWeights = lossWeights(:, Y2_new);

if size(lossWeights, 1) ~= K && size(lossWeights, 2) ~= N
    error('Error in Hamming loss function!');
end
end
