function loss = computeHammingLoss( result, gtImage, seedObj, seedBkg )

resultPixels = result( : );
gtPixels = gtImage( : );

if ~exist('seedObj', 'var')
    seedObj = false(size(gtImage,1), size(gtImage, 2));
end
if ~exist('seedBkg', 'var')
    seedBkg = false(size(gtImage,1), size(gtImage, 2));
end

resultPixels( seedObj(:) | seedBkg(:) ) = [];
gtPixels( seedObj(:) | seedBkg(:) ) = [];

nanMask = isnan(gtPixels);
resultPixels(nanMask) = [];
gtPixels(nanMask) = [];

loss = sum(resultPixels ~= gtPixels) / numel(gtPixels);

end

