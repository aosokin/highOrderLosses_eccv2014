function loss = computeHammingWeightedLoss( result, gtImage, objectWeights, seedObj, seedBkg )

resultPixels = result( : );
gtPixels = gtImage( : );
objectWeights = objectWeights( : );

if ~exist('seedObj', 'var')
    seedObj = false(size(gtImage,1), size(gtImage, 2));
end
if ~exist('seedBkg', 'var')
    seedBkg = false(size(gtImage,1), size(gtImage, 2));
end

resultPixels( seedObj(:) | seedBkg(:) ) = [];
gtPixels( seedObj(:) | seedBkg(:) ) = [];
objectWeights( seedObj(:) | seedBkg(:) ) = [];

nanMask = isnan(gtPixels);
resultPixels(nanMask) = [];
gtPixels(nanMask) = [];
objectWeights(nanMask) = [];

loss = sum(objectWeights .* (resultPixels ~= gtPixels)) / sum(objectWeights);

end

