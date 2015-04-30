function [F, C, warmStart] = oracle_graphCutSeeds_rowColumnLoss_normalized(iObject, Y, W, config)
%oracle_graphCutSeeds_rowColumnLoss_normalized - oracle for SSVM: loss augmented MAP-inference;
% binary segmentation, pairwise MRF, Potts pairwise weights, graphcut, user-defined seeds
% Silhouette loss normalized to have maximum of 1
% Separation oracle that finds maximum violated constraint
% W' * Psi(X, Y) >= L(W) - Xi
% L(W) is a linear function on W: L(W) = W' * F + C
% In undergenerating approach L(W) = W' * Psi(X, Y_worst) + loss(Y, Y_worst)
% Y_worst = argmax_(\bar{Y}) ( W' * Psi(X, \bar{Y}) + loss(\bar{Y}, Y) )
% object data is supposed to be stored in global variable X_dataset
%
% [F, C, warmStart] = oracle_graphCutSeeds_rowColumnLoss_normalized(iObject, Y, W, config);
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

if nargin ~= 4
    error([mfilename, ': Wrong number of input parameters']);
end
if nargout > 3
    error([mfilename, ': Wrong number of output parameters']);
end


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

%% construct the energy
% Multiply features by weights
uF = bsxfun(@times, unaryFeatures, unaryWeights);
pF = bsxfun(@times, pairwiseFeatures(:, 3 : end), pairwiseWeights');

% Compute data costs
dataCost = [zeros(nodeNum, 1), sum(uF, 1)'];

% incorporate seeds
seedWeight = max(max(abs(dataCost(:)) * 10000000), 10000000);
dataCost(objSeed(:) == 1, 1) = -seedWeight;
dataCost(bkgSeed(:) == 1, 2) = -seedWeight;

% construct silhouette sets to impose area constraints on them
nodeMask = reshape((1 : size(Y, 1) * size(Y, 2)), size(Y, 1), size(Y, 2));
constrSet = cell(size(Y, 1) + size(Y, 2), 1);
constrSetSize = size(constrSet, 1);
setNodeWeight = cell(size(Y, 1) + size(Y, 2), 1);
setWeight = ones(size(Y, 1) + size(Y, 2), 1);
for iRow = 1 : size(Y, 1)
    curSet = nodeMask(iRow, :)';
    constrSet{iRow} = curSet(~isnan(Y(curSet)));
    setNodeWeight{iRow} = ones(size(constrSet{iRow}));
end
for iCol = 1 : size(Y, 2)
    curSet = nodeMask(:, iCol);
    constrSet{size(Y, 1) + iCol} = curSet(~isnan(Y(curSet)));
    setNodeWeight{size(Y, 1) + iCol} = ones(size(constrSet{size(Y, 1) + iCol}));
end
% compute the object size in the ground truth 
sumYStar = nan(constrSetSize, 1);
for iSet = 1 : constrSetSize
    sumYStar(iSet) = sum(Y(constrSet{iSet}));
end

% Find the normalization factor for the loss
[variables, flagAdded] = loadVariableGlobalDataset('normalizationFactorRowColumn', iObject, config.loadDataInMemory);
if flagAdded
    normalizationFactor = variables{1};
else
    normalizationFactor = getMaxSetHammingLoss( Y, 1.0, constrSet, setNodeWeight, sumYStar, setWeight, objSeed, bkgSeed);
    addVariableGlobalDataset('normalizationFactorRowColumn', iObject, normalizationFactor);
end

% compute the maximum possible loss weight
lossWeight = config.distW / normalizationFactor;


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
    tmpNode = [(2 * lossWeight) * sumYStar(iSet), 0];  % add an extra node
    newTermWeights(endIndex + 1 : endIndex + size(tmpNode, 1), :) = tmpNode;
    endIndex = endIndex + size(tmpNode, 1);
    
    zIndexSet(iSet) = endIndex; % index of the new node
    newTermWeights(constrSet{iSet}, 1) = newTermWeights(constrSet{iSet}, 1) - 1 * lossWeight; % modify the unaries
end
newTermWeights(any(isnan(newTermWeights), 2), :) = [];
termWeights = newTermWeights;

%add edges related to the new node
newEdgeWeights = nan(5000000, 4);
newEdgeWeights(1 : size(edgeWeights, 1), :) = edgeWeights;
endIndex = size(edgeWeights, 1);
for iSet = 1 : constrSetSize
    curOnes = ones(length(constrSet{iSet}), 1);
    tmpEdges = [zIndexSet(iSet) * curOnes, constrSet{iSet}, (2 * lossWeight) * curOnes, 0 * curOnes];
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
    error([mfilename, ': error in graph cut']);
end
zValue = nodeLabels_worst(zIndexSet);
nodeLabels_worst(zIndexSet) = [];
Y_worst = nodeLabels_worst(nodeMap);

%% Analyze the results
% check if seed constraints are satisfied
if any(Y_worst(:) .* bkgSeed(:) > 0) || any((1 - Y_worst(:)) .* objSeed(:) > 0)
    warning(['MATLAB:', mfilename], [mfilename, ':ERROR IN ORACLE: seeds did not work!']);
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
    C = C - lossWeight * (2 * zValue(iSet) * (sumYStar(iSet) - sum(Y_worst(constrSet{iSet}))) + sum(Y_worst(constrSet{iSet})) - sumYStar(iSet));
end

% check if computed linear bound equals the energy computed by GraphCut
checkDiff = abs((-energy_worst + lossWeight * sum(sumYStar)) - (F' * W + C));
if  checkDiff > 1e-4
    warning(['MATLAB:', mfilename], [mfilename, ':ERROR IN ORACLE: computing the loss-augmented energy, error size: ', num2str(checkDiff)]);
end

tOracle = toc(tStart);
% fprintf('oracle time: %f\n', tOracle);
end

