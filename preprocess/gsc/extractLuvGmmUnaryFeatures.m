function features = extractLuvGmmUnaryFeatures( image, labelImg, gmmComponentNumber )

luvImage = RGB2Luv(image);

points = reshape(luvImage, [size(image, 1) * size(image, 2), 3]);
features = points;

points = bsxfun(@rdivide, points, max(points))';

for iGmm = 1 : length(gmmComponentNumber)
    segOpts = struct;
    segOpts.gmmNmix_bg = gmmComponentNumber(iGmm);
    segOpts.gmmNmix_fg = gmmComponentNumber(iGmm);
    segOpts.posteriorMethod = 'gmm_bs_mixtured';
    segOpts.gmmUni_value = 1; % assuming features in [0,1]
    segOpts.gmmLikeli_gamma = 0.05;
    
    posteriorImage = getPosteriorImage(points, labelImg, segOpts);

    posteriorImage = posteriorImage(:);
    prob_densities = [-log(1 - posteriorImage), -log(posteriorImage)];
    prob_densities(prob_densities > 100) = 100;
    
    features = [features, prob_densities];
end

end

