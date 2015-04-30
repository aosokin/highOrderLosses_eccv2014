function [F, C, warmStart] = oracle_graphCutSeeds_areaLoss_normalized(iObject, Y, W, config)
%oracle_graphCutSeeds_areaLoss_normalized - oracle for SSVM: loss augmented MAP-inference;
% binary segmentation, pairwise MRF, Potts pairwise weights, graphcut, user-defined seeds
% Area loss normalized to have maximum of 1
% Separation oracle that finds maximum violated constraint
% W' * Psi(X, Y) >= L(W) - Xi
% L(W) is a linear function on W: L(W) = W' * F + C
% In undergenerating approach L(W) = W' * Psi(X, Y_worst) + loss(Y, Y_worst)
% Y_worst = argmax_(\bar{Y}) ( W' * Psi(X, \bar{Y}) + loss(\bar{Y}, Y) )
% object data is supposed to be stored in global variable X_dataset
%
% [F, C, warmStart] = oracle_graphCutSeeds_areaLoss_normalized(iObject, Y, W, config);
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
%       epsNormalization - constant to adjust thresholds for checks
%       
% OUTPUT
%   F - coefficient; double numWeights x 1;
%   C - constant term in linear constraint; double 1 x 1;
%   warmStart - warm start for the next oracle on the same object (if required);
%
%   Anton Osokin, 08.12.2012

if nargin ~= 4
    error([mfilename, ': Wrong number of input parameters']);
end
if nargout > 3
    error([mfilename, ': Wrong number of output parameters']);
end

% fprintf('Starting GraphCut oracle: ');
tStart = tic;

% This oracle does not have a warmStart options
warmStart = [];

% Number of classes
K = config.K;

% check for back compatability
if ~isfield(config, 'epsNormalization')
    config.epsNormalization = 1;
end

% Extract unary features
unaryWeights = W(1 : config.unaryFeatureNum);
% Extract pairwise features
pairwiseWeights = W(config.unaryFeatureNum + 1 : config.unaryFeatureNum + config.pairwiseFeatureNum);

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

% Multiply features by weights
uF = bsxfun(@times, unaryFeatures, unaryWeights);
pF = bsxfun(@times, pairwiseFeatures(:, 3 : end), pairwiseWeights');

% Compute data costs
dataCost = [zeros(nodeNum, 1), sum(uF, 1)'];

% incorporate seeds
seedWeight = max(max(abs(dataCost(:)) * 10000000 + config.distW * 10 ), 10000000);
dataCost(objSeed(:) == 1, 1) = -seedWeight;
dataCost(bkgSeed(:) == 1, 2) = -seedWeight;

% Find the normalization factor for the loss
[variables, flagAdded] = loadVariableGlobalDataset('normalizationFactorArea', iObject, config.loadDataInMemory);
if flagAdded
    normalizationFactor = variables{1};
else
    effictiveGT = Y(:);
    effictiveGT = effictiveGT(objSeed(:) == 0 & bkgSeed(:) == 0 & ~isnan(Y(:)));
    normalizationFactor = max( sum(effictiveGT), length(effictiveGT) - sum(effictiveGT) );
    addVariableGlobalDataset('normalizationFactorArea', iObject, normalizationFactor);
end


% find the weight of the loss
lossWeight = config.distW / normalizationFactor;

% augment energy with a loss function
nodeMask = (1 : size(Y, 1) * size(Y, 2))';
nodeMask = nodeMask(~isnan(Y(:)));
sumYStar = sum(Y(nodeMask));
dataCost = [dataCost; 0, lossWeight * (-2 * sumYStar) ]; % add an extra node
zIndex = size(dataCost, 1); % index of the new node
dataCost(nodeMask, 2) = dataCost(nodeMask, 2) + 1 * lossWeight; % modify the unaries

% Build a graph
termWeights = double([-dataCost(:, 2), -dataCost(:, 1)]);
tmp = sum(pF, 2);
tmp(tmp > 0) = 0;
edgeWeights = [pairwiseFeatures(:, 1), pairwiseFeatures(:, 2), -tmp, -tmp];

%add edges related to the new node
curOnes = ones(length(nodeMask), 1);
edgeWeights = [edgeWeights; zIndex * curOnes, nodeMask, (2 * lossWeight) * curOnes, 0 * curOnes];

edgeWeights(edgeWeights(:, 3) < 0, 3) = 0;
edgeWeights(edgeWeights(:, 4) < 0, 4) = 0;

% Run GraphCut
try
    [energy_worst, nodeLabels_worst] = graphCutMex(termWeights, edgeWeights);
catch err
    save('error_graphCut.mat', 'termWeights', 'edgeWeights');
    error([mfilename, ': error in graph cut']);
end
zValue = nodeLabels_worst(zIndex);
nodeLabels_worst(zIndex) = [];
Y_worst = nodeLabels_worst(nodeMap);

if any(Y_worst(:) .* bkgSeed(:) > 0) || any((1 - Y_worst(:)) .* objSeed(:) > 0)
    error([mfilename, ':ERROR IN ORACLE: seeds did not work!']);
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
C = - lossWeight * (2 * zValue * (sumYStar - sum(Y_worst(nodeMask))) + sum(Y_worst(nodeMask)) -sumYStar);

checkDiff = abs((-energy_worst + lossWeight * sumYStar) - (F' * W + C));
if  checkDiff > 1e-2 * config.epsNormalization
    warning([mfilename, ':ERROR IN ORACLE: computing the loss-augmented energy, error size: ', num2str(checkDiff)]);
end

tOracle = toc(tStart);
% fprintf('oracle time: %f\n', tOracle);
end

