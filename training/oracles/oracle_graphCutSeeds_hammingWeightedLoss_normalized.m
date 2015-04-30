function [F, C, warmStart, Y_worst] = oracle_graphCutSeeds_hammingWeightedLoss_normalized(iObject, Y, W, config)
%oracle_graphCutSeeds_hammingWeightedLoss_normalized - oracle for SSVM: loss augmented MAP-inference;
% binary segmentation, pairwise MRF, Potts pairwise weights, graphcut, user-defined seeds
% weighted Hamming loss normalized to have maximum of 1
% Separation oracle that finds maximum violated constraint
% W' * Psi(X, Y) >= L(W) - Xi
% L(W) is a linear function on W: L(W) = W' * F + C
% In undergenerating approach L(W) = W' * Psi(X, Y_worst) + loss(Y, Y_worst)
% Y_worst = argmax_(\bar{Y}) ( W' * Psi(X, \bar{Y}) + loss(\bar{Y}, Y) )
% object data is supposed to be stored in global variable X_dataset
%
% [F, C, warmStart] = oracle_graphCutSeeds_hammingWeightedLoss_normalized(iObject, Y, W, config);
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


% fprintf('Starting GraphCut oracle: ');
tStart = tic;

% This oracle does not have a warmStart options
warmStart = [];

% Number of classes
K = config.K;

% Extract unary features
unaryWeights = W(1 : config.unaryFeatureNum);
% Extract pairwise features
pairwiseWeights = W(config.unaryFeatureNum + 1 : config.unaryFeatureNum + config.pairwiseFeatureNum);

% load data
requiredVariablesList = {'unaryFeatures', 'pairwiseFeatures', 'nodeMap', 'objSeed', 'bkgSeed', 'hammingWeights'};
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
hammingWeights = double(variables{6});

% Multiply features by weights
uF = bsxfun(@times, unaryFeatures, unaryWeights);
pF = bsxfun(@times, pairwiseFeatures(:, 3 : end), pairwiseWeights');

% Compute data costs
dataCost = [zeros(nodeNum, 1), sum(uF, 1)'];

% incorporate seeds
seedWeight = max(max(abs(dataCost(:)) * 10000000), 10000000);
dataCost(objSeed(:) == 1, 1) = -seedWeight;
dataCost(bkgSeed(:) == 1, 2) = -seedWeight;

% Find the normalization factor for the loss
[variables, flagAdded] = loadVariableGlobalDataset('normalizationFactorHammingWeighted', iObject, config.loadDataInMemory);
if flagAdded
    normalizationFactor = variables{1};
else
    normalizationFactor = sum( hammingWeights(:) .* (objSeed(:) == 0 & bkgSeed(:) == 0 & ~isnan(Y(:))) );
    addVariableGlobalDataset('normalizationFactorHammingWeighted', iObject, normalizationFactor);
end

% Update data costs with loss function
lossWeights = lossWeights_Hamming(Y(:));
lossWeights = bsxfun(@times, lossWeights, hammingWeights(:)'); % update the loss weights with weight Map
lossWeights = lossWeights / normalizationFactor * config.distW;

dataCost(:, 1) = dataCost(:, 1) + accumarray(nodeMap(:), lossWeights(1, :)', [nodeNum 1], @sum, 0);
dataCost(:, 2) = dataCost(:, 2) + accumarray(nodeMap(:), lossWeights(2, :)', [nodeNum 1], @sum, 0);

% Build a graph
termWeights = double([-dataCost(:, 2), -dataCost(:, 1)]);
tmp = sum(pF, 2);
tmp(tmp > 0) = 0;
edgeWeights = [pairwiseFeatures(:, 1), pairwiseFeatures(:, 2), -tmp, -tmp];

% Run GraphCut
try
    [energy, nodeLabels_worst] = graphCutMex(termWeights, edgeWeights);
catch err
    save('error_graphCut.mat', 'termWeights', 'edgeWeights');
    error([mfilename, ': error in graph cut']);
end
Y_worst = nodeLabels_worst(nodeMap);

if any(Y_worst(:) .* bkgSeed(:) > 0) || any((1 - Y_worst(:)) .* objSeed(:) > 0)
    error([mfilename, ': ERROR IN ORACLE: seeds did not work!']);
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
%C = loss(Y_worst(:) + 1, Y(:) + 1, config);
C = sum(lossWeights(Y_worst(:) + 1 + ((1 : nodeNum)' - 1) * K));

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
