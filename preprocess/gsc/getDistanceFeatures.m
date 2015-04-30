function features = getDistanceFeatures(image, seedMask)

featureNumber = 600;
features = nan(numel(seedMask), featureNumber);
lastFeatureIndex = 0;

thresholdLevelNumber = 5;

% compute Euclidian distances:
objSeedEuclDist = getGeodesicDistance(seedMask == 1, 4, image, 0);
bkgSeedEuclDist = getGeodesicDistance(seedMask == 2, 4, image, 0);


maxObj = max(objSeedEuclDist(:));
for iLevel = 1 : thresholdLevelNumber;
    tmp = objSeedEuclDist(:);
    tmp = tmp - iLevel / thresholdLevelNumber * maxObj;
    tmp(tmp > 0) = 0;
            
    lastFeatureIndex = lastFeatureIndex + 1;
    features(:, lastFeatureIndex) = tmp;
end
        
maxBkg = max(bkgSeedEuclDist(:));
for iLevel = 1 : thresholdLevelNumber;
    tmp = bkgSeedEuclDist(:);
    tmp = tmp - iLevel / thresholdLevelNumber * maxBkg;
    tmp(tmp > 0) = 0;
            
    lastFeatureIndex = lastFeatureIndex + 1;
    features(:, lastFeatureIndex) = tmp;
end

% compute Luv dist:
geoGamma = [1.0];%[0.25; 0.5; 0.75; 1.0];
colorDistancePower = [2];

luvImage = RGB2Luv(image);
objSeedLuvDist = cell(length(geoGamma), length(colorDistancePower));
bkgSeedLuvDist = cell(length(geoGamma), length(colorDistancePower));

for iGamma = 1 : length(geoGamma)
    for iPower = 1 : length(colorDistancePower);
        
        opts = struct;
        opts.geoGamma = geoGamma(iGamma);
        opts.colorDistancePower = colorDistancePower( iPower );
        opts.colorDistanceMax = 100;
    
        objSeedLuvDist{iGamma, iPower} = getGeodesicDistance(seedMask == 1, 8, luvImage, opts);
        bkgSeedLuvDist{iGamma, iPower} = getGeodesicDistance(seedMask == 2, 8, luvImage, opts);
        
        
        
        maxObj = max(objSeedLuvDist{iGamma}(:));
        for iLevel = 1 : thresholdLevelNumber;
            tmp = objSeedLuvDist{iGamma}(:);
            tmp = tmp - iLevel / thresholdLevelNumber * maxObj;
            tmp(tmp > 0) = 0;
            
            lastFeatureIndex = lastFeatureIndex + 1;
            features(:, lastFeatureIndex) = tmp;
        end
        
        maxBkg = max(bkgSeedLuvDist{iGamma}(:));
        for iLevel = 1 : thresholdLevelNumber;
            tmp = bkgSeedLuvDist{iGamma}(:);
            tmp = tmp - iLevel / thresholdLevelNumber * maxBkg;
            tmp(tmp > 0) = 0;
            
            lastFeatureIndex = lastFeatureIndex + 1;
            features(:, lastFeatureIndex) = tmp;
        end
        
    end
end

% compute RGB dist:
geoGamma = [1.0];%[0.25; 0.5; 0.75; 1.0];
colorDistancePower = [2];

for iGamma = 1 : length(geoGamma)
    for iPower = 1 : length(colorDistancePower);
        
        opts = struct;
        opts.geoGamma = geoGamma(iGamma);
        opts.colorDistancePower = colorDistancePower( iPower );
        opts.colorDistanceMax = 0.3;
    
        objSeedLuvDist{iGamma} = getGeodesicDistance(seedMask == 1, 8, image, opts);
        bkgSeedLuvDist{iGamma} = getGeodesicDistance(seedMask == 2, 8, image, opts);
        
        
        
        maxObj = max(objSeedLuvDist{iGamma}(:));
        for iLevel = 1 : thresholdLevelNumber;
            tmp = objSeedLuvDist{iGamma}(:);
            tmp = tmp - iLevel / thresholdLevelNumber * maxObj;
            tmp(tmp > 0) = 0;
            
            lastFeatureIndex = lastFeatureIndex + 1;
            features(:, lastFeatureIndex) = tmp;
        end
        
        maxBkg = max(bkgSeedLuvDist{iGamma}(:));
        for iLevel = 1 : thresholdLevelNumber;
            tmp = bkgSeedLuvDist{iGamma}(:);
            tmp = tmp - iLevel / thresholdLevelNumber * maxBkg;
            tmp(tmp > 0) = 0;
            
            lastFeatureIndex = lastFeatureIndex + 1;
            features(:, lastFeatureIndex) = tmp;
        end
        
    end
end

% compute posterior dist:
geoGamma = [1.0];%[0.25; 0.5; 0.75; 1.0];
colorDistancePower = [2];

posteriorImage = getPosteriorImageTotal(image, seedMask);
for iGamma = 1 : length(geoGamma)
    for iPower = 1 : length(colorDistancePower);
        
        opts = struct;
        opts.geoGamma = geoGamma(iGamma);
        opts.colorDistancePower = colorDistancePower( iPower );
        opts.colorDistanceMax = 1.0;
    
        objSeedLuvDist{iGamma} = getGeodesicDistance(seedMask == 1, 8, posteriorImage, opts);
        bkgSeedLuvDist{iGamma} = getGeodesicDistance(seedMask == 2, 8, posteriorImage, opts);
        
        
        
        maxObj = max(objSeedLuvDist{iGamma}(:));
        for iLevel = 1 : thresholdLevelNumber;
            tmp = objSeedLuvDist{iGamma}(:);
            tmp = tmp - iLevel / thresholdLevelNumber * maxObj;
            tmp(tmp > 0) = 0;
            
            lastFeatureIndex = lastFeatureIndex + 1;
            features(:, lastFeatureIndex) = tmp;
        end
        
        maxBkg = max(bkgSeedLuvDist{iGamma}(:));
        for iLevel = 1 : thresholdLevelNumber;
            tmp = bkgSeedLuvDist{iGamma}(:);
            tmp = tmp - iLevel / thresholdLevelNumber * maxBkg;
            tmp(tmp > 0) = 0;
            
            lastFeatureIndex = lastFeatureIndex + 1;
            features(:, lastFeatureIndex) = tmp;
        end
        
    end
end


colMask = any(isnan(features));
features(:, colMask) = [];

end