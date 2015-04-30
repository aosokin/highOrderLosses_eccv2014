function example = preprocessImage(filename, data_dir, options)
% preprocessImage computes the features of interest for an image
%
% The initial version of this function was provided by Patrick Pletscher in the HOL package:
% https://github.com/ppletscher/hol
%
% This function was modified by Anton Osokin


options_default = defaultOptions();
if (nargin >= 3)
    options = processOptions(options, options_default);
else
    options = options_default;
end

example = [];

fname = [data_dir 'images/' filename];
img = imread(fname);
img = im2double(img);
fname = [data_dir 'images-gt/' filename(1:end-4) '.png'];

% AOSOKIN
imageLabel = im2double(imread(fname));
label = nan(size(imageLabel, 1), size(imageLabel, 2));
label(imageLabel > 0.9) = 1;
label(imageLabel < 0.1) = 0;


if ~isfield( options,  'seedDir')
    fname = [data_dir 'images-labels/' filename(1:end-4) '-anno.png'];
    label_seed = imread(fname);
else
    fname = fullfile(options.seedDir, [ filename(1:end-4), '-anno.png']);
    label_seed = imread(fname);
end



example.image = img;
example.label = label;
example.label_seed = label_seed;


opts = [];
opts.posteriorMethod = 'gmm_bs_mixtured';
opts.gmmUni_value=1; % assuming features in [0,1]
opts.gmmLikeli_gamma=0.05;
opts.featureSpace = 'rgb';
opts.gcGamma = 150;
opts.gcSigma_c = 'auto';
opts.gcScale = 50;
%opts.gcNbrType = 'colourGradient';
%opts.gcNbrType = 'colourGradient8';
opts.gcNbrType = options.connectivity;

[features, edges, edge_weights, edge_types, edgeColorWeights] = extractFeatures(img, opts);
example.features = features;
example.edge_weights = edge_weights;
example.edgeColorWeights = edgeColorWeights;

% AOSOKIN: add different number of Gaussians
for iMixtureSize = 1 : length(options.numberOfGaussians)
    opts.gmmNmix_fg = options.numberOfGaussians(iMixtureSize); % AOSOKIN
    opts.gmmNmix_bg = options.numberOfGaussians(iMixtureSize); % AOSOKIN
    posterior_image = getPosteriorImage(features, label_seed, opts);
    example.posterior_image{iMixtureSize} = posterior_image;

    posterior_image = posterior_image(:)';
    bg_clamp=(label_seed(:)==2);
    fg_clamp=(label_seed(:)==1);

    prob_densities=[-log(1-posterior_image); -log(posterior_image)];
    prob_densities(prob_densities>100)=100;

    % TODO: check again which values to use??
    % prob_densities(2,bg_clamp)=inf;prob_densities(1,bg_clamp)=0;
    % prob_densities(1,fg_clamp)=inf;prob_densities(2,fg_clamp)=0;
    % AOSOKIN: seeds are taken care in other fashion
    % prob_densities(2,bg_clamp)=2000;prob_densities(1,bg_clamp)=0;
    % prob_densities(1,fg_clamp)=2000;prob_densities(2,fg_clamp)=0;

    prob_densities=prob_densities*opts.gcScale;
    example.prob_densities{iMixtureSize} = prob_densities;
end

% opts=gscSeq.segOpts();
% segH=gscSeq.segEngine(0,opts);
% segH.preProcess(im2double(img));
% ok=segH.start(example.label_seed);
% gscSeq_label = double(segH.seg>0);

% TODO: check whether we should reenable this!
%example.features_unary = [example.features; double(example.prob_densities); gscSeq_label(:)'; ones(1, size(example.features,2))];
example.features_unary = [example.features; double(cat(1, example.prob_densities{:})); ones(1, size(example.features,2))];

example.edges = edges;
example.edges_type = edge_types;
if (~options.different_edge_type)
    example.edges_type = ones(1, size(example.edges,2));
end
example.features_pairwise = [double(edge_weights(:)'); ones(1,numel(edge_weights))];


function options = defaultOptions()

options = [];
options.connectivity = 'colourGradient';
options.different_edge_type = 0;
options.numberOfGaussians = 5;

