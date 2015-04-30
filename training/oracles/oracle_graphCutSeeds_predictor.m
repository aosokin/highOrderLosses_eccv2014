function Y = oracle_graphCutSeeds_predictor(X, W, config)
%oracle_graphCutSeeds_predictor is an oracle that finds a best labeling
% Y = argmax_Y  W' * Psi(X, Y)
% Input: object X, weights W, config
% Output: labels Y

tStart = tic;

% Number of classes
K = 2;
if (config.K ~= 2)
    error('graphCut oracle can work only with 2 classes');
end

% Extract unary features
unaryWeights = W(1 : config.unaryFeatureNum);
% Extract pairwise features
pairwiseWeights = W(config.unaryFeatureNum + 1 : config.unaryFeatureNum + config.pairwiseFeatureNum);

% load data
if isfield(X, 'unaryFeatures') && isfield(X, 'pairwiseFeatures') && isfield(X, 'nodeMap') && isfield(X, 'objSeed') && isfield(X, 'bkgSeed') 
   unaryFeatures = X.unaryFeatures;
   pairwiseFeatures = X.pairwiseFeatures;
   nodeMap = X.nodeMap;
   objSeed = X.objSeed;
   bkgSeed = X.bkgSeed;
else
    load(X.dataFile, 'unaryFeatures', 'pairwiseFeatures', 'nodeMap', 'objSeed', 'bkgSeed');
end
nodeNum = max(nodeMap(:));
if size(unaryFeatures, 1) ~= config.unaryFeatureNum || size(unaryFeatures, 2) ~= nodeNum
    error(['sSVM:Oracle_GC: Error in unaryFeatures in file ', X.dataFile]);
end

% Multiply features by weights
uF = bsxfun(@times, unaryFeatures, unaryWeights);
pF = bsxfun(@times, pairwiseFeatures(:, 3 : end), pairwiseWeights');

% Compute data costs
dataCost = [zeros(nodeNum, 1), sum(uF, 1)'];

% incorporate seeds
seedWeight = max(max(abs(dataCost(:)) * 1000), 1000);
dataCost(objSeed(:) == 1, 1) = -seedWeight;
dataCost(bkgSeed(:) == 1, 2) = -seedWeight;

% Build a graph
termWeights = double([-dataCost(:, 2), -dataCost(:, 1)]);
tmp = sum(pF, 2);
tmp(tmp > 0) = 0;
edgeWeights = double([pairwiseFeatures(:, 1), pairwiseFeatures(:, 2), -tmp, -tmp]);

% Run GraphCut
try
    [~, nodeLabels_worst] = graphCutMex(termWeights, edgeWeights);
catch err
    save('error_graphCut.mat', 'termWeights', 'edgeWeights');
    error('sSVM:oracle: error in graph cut');
end
Y = nodeLabels_worst(nodeMap);

tOracle = toc(tStart);
% fprintf('oracle time: %f\n', tOracle);
end
