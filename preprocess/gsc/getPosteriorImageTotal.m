function posteriorImage = getPosteriorImageTotal(image, seeds)

opts = struct;
opts.gmmNmix_fg=5;
opts.gmmNmix_bg=5;
opts.gmmUni_value=1; % assuming features in [0,1]
opts.gmmLikeli_gamma=0.05;
opts.posteriorMethod='gmm_bs_mixtured';
opts.featureSpace='rgb';

features = reshape(image, [size(image, 1) * size(image, 2), 3]);
features = features';

posteriorImage = getPosteriorImage(features, seeds, opts);

end
