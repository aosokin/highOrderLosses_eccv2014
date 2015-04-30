function loss = computeJaccardLoss( result, gtImage, seedObj, seedBkg )

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

TP = sum( resultPixels == 1 & gtPixels == 1);
FP = sum( resultPixels == 1 & gtPixels == 0);
FN = sum( resultPixels == 0 & gtPixels == 1);

loss = (1 - TP / ( TP + FP + FN) );

end

