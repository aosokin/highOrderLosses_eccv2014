function [maxLoss, worstLabeling] = getMaxSetHammingLoss(gtImage, alphaWeight,  setNodeIndex, setNodeWeight, setSum, setWeight, objSeed, bkgSeed )
%getMaxSetHammingLoss function computes the maximum possible value of the following loss (w.r.t. Y):
%   \Delta( Y, Y_true ) = \alpha * \Delta_Set(Y, Y_true) + (1 - \alpha) * \Delta_Hamming(Y, Y_true)
% where \alpha is defined by alphaWeight,
%       \Delta_Set - by setNodeIndex, setNodeWeight, setSum, setWeight
%       Y_true - by gtImage

if ~exist('objSeed', 'var')
    objSeed = false(size(gtImage,1), size(gtImage, 2));
end
if ~exist('bkgSeed', 'var')
    bkgSeed = false(size(gtImage,1), size(gtImage, 2));
end

nodeNum = size(gtImage, 1) * size(gtImage, 2);

% incorporate Hamming loss
lossWeights = lossWeights_Hamming( gtImage(:) ) * (1 - alphaWeight);

%% Build a graph
termWeights = double([-lossWeights(2, :)', -lossWeights(1, :)']);
edgeWeights = nan(0, 4);

% incorporate seeds
seedWeight = max(max(abs(termWeights(:)) * 10000000 + 10 ), 10000000);
termWeights(objSeed(:) == 1, 2) = seedWeight;
termWeights(bkgSeed(:) == 1, 1) = seedWeight;

%% augment energy with a loss function
% add new unary nodes and their unary potentials
lossSkeletonWeight = alphaWeight;
constrSetSize = length(setNodeIndex);
zIndexSet = nan(constrSetSize, 1);
newTermWeights = nan(1000000, 2);
newTermWeights(1 : size(termWeights, 1), :) = termWeights;
endIndex = size(termWeights, 1);
for iSet = 1 : constrSetSize
    tmpNode = [(2 * lossSkeletonWeight) * setSum(iSet) * setWeight(iSet), 0];  % add an extra node
    newTermWeights(endIndex + 1, :) = tmpNode;
    endIndex = endIndex + 1;

    zIndexSet(iSet) = endIndex; % index of the new node
    newTermWeights(setNodeIndex{iSet}, 1) = newTermWeights(setNodeIndex{iSet}, 1) - setNodeWeight{iSet} * lossSkeletonWeight * setWeight(iSet); % modify the unaries
end
newTermWeights(any(isnan(newTermWeights), 2), :) = [];
termWeights = newTermWeights;

%add edges related to the new node
newEdgeWeights = nan(5000000, 4);
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

%constant
constant = 0;
for iSet = 1 : constrSetSize
    constant = constant - lossSkeletonWeight * setSum(iSet) * setWeight(iSet);
end
    

%% Run GraphCut
try
    [energy_worst, nodeLabels_worst] = graphCutMex(termWeights, edgeWeights);
catch err
    save('error_graphCut.mat', 'termWeights', 'edgeWeights');
    error('getMaxSetHammingLoss:graphCutFailed', 'error in graph cut');
end
maxLoss = -energy_worst - constant;
worstLabeling = reshape( nodeLabels_worst(1 : nodeNum), [size(gtImage, 1), size(gtImage, 2)] );

% check if seed constraints are satisfied
if any(worstLabeling(:) .* bkgSeed(:) > 0) || any((1 - worstLabeling(:)) .* objSeed(:) > 0)
    warning('ERROR IN ORACLE: seeds did not work!');
end

end

function lossWeights = lossWeights_Hamming(Y2)
% Hamming loss between Y1 and Y2
% loss should be equal to sum_i sum_k [Y1(i) = k] * lossWeights(k, i)
% Y2 is a vector that contains 0, 1, NaN
% for NaN positions loss should be always zero

% Get parameters
K = 2;
N = sum(~isnan(Y2));

Y2_new = Y2 + 1;
Y2_new(isnan(Y2)) = K + 1;

% Compute lossWeights
lossWeights = [1 - eye(K), zeros(K, 1)] / N;

lossWeights = lossWeights(:, Y2_new);

if size(lossWeights, 1) ~= K && size(lossWeights, 2) ~= N
    error('getMaxSetHammingLoss:lossWeights_Hamming:wrongOutputSizes','error in Hamming loss function!');
end
end
