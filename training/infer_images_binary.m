function [Y] = infer_images_binary(X, W, options)
%infer_images_binary perfoms MAP-inference an a dataset
% 
% Y = infer_images_binary(X, W, options);
% 
% INPUT
%   X - cell array of objects;
%   W - vector with learned weights
%   options - structure with fields:
%       classNum - number of classes
%       unaryFeatureNum - number of unary features
%       pairwiseFeatureNum - number of pairwise features
%       predictor - oracle-type function for predicting an answer for each particular object
%
% OUTPUT
%   Y - cell array of labeling;
%
%   Anton Osokin, 27.11.2012



%% Initialization

% number of objects
config.N = length(X);

% number of classes
config.K = options.classNum;

% number of unary features
config.unaryFeatureNum = options.unaryFeatureNum;

% number of pairwise features
config.pairwiseFeatureNum = options.pairwiseFeatureNum;

% number of weights
if config.K == 2
    config.M = config.unaryFeatureNum + config.pairwiseFeatureNum;
else
    config.M = config.unaryFeatureNum * config.K + config.pairwiseFeatureNum;
end

if(length(W) ~= config.M)
    error('Inconsistent number of features');
end

Y = cell(config.N, 1);
for i = 1 : config.N
    fprintf('Working with object %d: ', i);
    tStart = tic;
    Y{i} = options.predictor(X{i}, W, config);
    fprintf('%f seconds\n', toc(tStart));
end

end