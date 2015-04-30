function f = psi_images_binary(iObject, Y, config)
%psi_images_binary computes the joint feature map for object iObject and labeling Y
%
% f = psi_images_binary(iObject, Y, config);
%
% INPUT
%   iObject - index of object X, integer 1 x 1'
%   Y - labeling Y; double imageHeight x imageWidth;
%   config - structure od method parameters:
%       K - number of classes
%       unaryFeatureNum - number fo unary features
%       pairwiseFeatureNum - number of pairwise features
%       
% OUTPUT
%   f - the feature map
%
%   Anton Osokin, 08.12.2012

% load data form file X.dataFile or from fields of X
requiredVariablesList = {'unaryFeatures', 'pairwiseFeatures', 'nodeMap'};
[variables, variablesLoaded] = loadVariableGlobalDataset(requiredVariablesList, iObject, config.loadDataInMemory);
if any(~variablesLoaded)
    badVarStr = strjoin(requiredVariablesList(~variablesLoaded), '; ');
    error([mfilename,':dataNotLoaded'], ['Could not load ', badVarStr, ' for object #', num2str(iObject)]);
end
unaryFeatures = double(variables{1});
pairwiseFeatures = double(variables{2});
nodeMap = double(variables{3});
nodeNum = max(nodeMap(:));

if size(unaryFeatures, 1) ~= config.unaryFeatureNum || size(unaryFeatures, 2) ~= nodeNum
    error([mfilename,':badUnaryFeatures'], ['Error in unaryFeatures in object ', num2str(iObject)]);
end
if size(nodeMap, 1) ~= size(Y, 1) || size(nodeMap, 2) ~= size(Y, 2)
    error([mfilename,':badY'], ['Error in Y: ', num2str(iObject)]);
end

% compute the node labeling
nodeSize = accumarray(nodeMap(:), 1, [nodeNum 1], @sum, 0); 
nodeLabel = double(accumarray(nodeMap(:), Y(:) == 1, [nodeNum 1], @sum, 0) ./ nodeSize > 0.5);

% compute unary features
uF = sum(bsxfun(@times, unaryFeatures, nodeLabel'), 2);

% compute pairwise features
activePotts = (nodeLabel(pairwiseFeatures(:, 1)) ~= nodeLabel(pairwiseFeatures(:, 2)));
pF = sum(pairwiseFeatures(activePotts, 3 : end), 1)';

f = [uF; pF];

end
