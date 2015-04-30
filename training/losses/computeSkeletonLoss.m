function fullLoss = computeSkeletonLoss( resultImage, gtImage, objSeed, bkgSeed )

if ~exist('objSeed', 'var')
    objSeed = false(size(gtImage,1), size(gtImage, 2));
end
if ~exist('bkgSeed', 'var')
    bkgSeed = false(size(gtImage,1), size(gtImage, 2));
end

% construct sets for the loss
xPos = repmat( 1 : size(gtImage, 2), [size(gtImage, 1), 1] );
yPos = repmat( (1 : size(gtImage, 1))', [1, size(gtImage, 2)] );

% construct sets from skeleton to the boundary
boundary = bwmorph(gtImage, 'remove');
skeleton = bwmorph(gtImage, 'skel', inf);
skeletonPoints = find(skeleton(:));
distToBoundary = bwdist(boundary);

% decrease the number of skeleton points
skeletonPoints = skeletonPoints(1 : 4 : end);
radiusWeights = [1.25, 1.25 * 0.5, 1.25 * 0.25];
constrSetSize = length(skeletonPoints) * length(radiusWeights);
setNodeIndex = cell(constrSetSize, 1);
setNodeWeight = cell(constrSetSize, 1);
toDelete = true(constrSetSize, 1);
goodPoints = ~isnan(gtImage) & ~objSeed & ~bkgSeed;
iSet = 0;
for iPoint = 1 : length(skeletonPoints)
    xCenter = xPos( skeletonPoints(iPoint) );
    yCenter = yPos( skeletonPoints(iPoint) );
    curDist = distToBoundary( skeletonPoints(iPoint) );
    for iRadius = 1 : length(radiusWeights)
        radius = curDist * radiusWeights(iRadius);
        mask = (xPos - xCenter).^2 + (yPos - yCenter).^2 < radius ^ 2;
        mask = mask & goodPoints;
    
        iSet = iSet + 1;
        setNodeIndex{iSet} = find(mask(:));
    
        if numel(setNodeIndex{iSet}) < 1
            toDelete(iSet) = true;
        else
            toDelete(iSet) = false;
        end
    
        setNodeWeight{iSet} = ones(length(setNodeIndex{iSet}), 1);
    end
end

setNodeIndex(toDelete) = [];
setNodeWeight(toDelete) = [];
constrSetSize = length(setNodeIndex);

setWeight = nan(constrSetSize, 1);
setSum = nan(constrSetSize, 1);
for iSet = 1 : constrSetSize
    setSum(iSet) = sum(gtImage(setNodeIndex{iSet}) .* setNodeWeight{iSet} );
    setWeight(iSet) = 1 / numel(setNodeIndex{iSet});
end

[maxSkeletonLoss, badY] = getMaxSetHammingLoss( gtImage, 1.0, setNodeIndex, setNodeWeight, setSum, setWeight, objSeed, bkgSeed);

% aggregate sets to compute the loss
lossSkeleton = 0;
for iSet = 1 : constrSetSize
    setWeight(iSet) = setWeight(iSet) / maxSkeletonLoss;
    lossSkeleton = lossSkeleton + setWeight(iSet) * abs( sum(resultImage(setNodeIndex{iSet}) .* setNodeWeight{iSet}) - setSum(iSet) );
end

[maxFullLoss, badY] = getMaxSetHammingLoss( gtImage, 0.5, setNodeIndex, setNodeWeight, setSum, setWeight, objSeed, bkgSeed);

fullLoss = 0.5 * ( lossSkeleton + computeHammingLoss( resultImage, gtImage ));
fullLoss = fullLoss / maxFullLoss;

end
