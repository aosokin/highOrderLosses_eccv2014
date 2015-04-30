function loss = computeRowColumnLoss( resultImage, gtImage, objSeed, bkgSeed )

if ~exist('objSeed', 'var')
    objSeed = false(size(gtImage,1), size(gtImage, 2));
end
if ~exist('bkgSeed', 'var')
    bkgSeed = false(size(gtImage,1), size(gtImage, 2));
end


% construct sets for the loss
xPos = repmat( 1 : size(gtImage, 2), [size(gtImage, 1), 1] );
yPos = repmat( (1 : size(gtImage, 1))', [1, size(gtImage, 2)] );

% construct sets from rows and columns
constrSetSize = size(gtImage, 1) + size(gtImage, 2);
setNodeIndex = cell(constrSetSize, 1);
setNodeWeight = cell(constrSetSize, 1);
toDelete = true(constrSetSize, 1);
goodPoints = ~isnan(gtImage) & ~objSeed & ~bkgSeed;
iSet = 0;
for iRow = 1 : size(gtImage, 1)
    iSet = iSet + 1;
    mask = (yPos  == iRow);
    mask = mask & goodPoints;
    setNodeIndex{iSet} = find(mask(:));
    if numel(setNodeIndex{iSet}) < 1
        toDelete(iSet) = true;
    else
        toDelete(iSet) = false;
    end
    setNodeWeight{iSet} = ones(length(setNodeIndex{iSet}), 1);
end
for iCol = 1 : size(gtImage, 2)
    iSet = iSet + 1;
    mask = (xPos  == iCol);
    mask = mask & goodPoints;
    setNodeIndex{iSet} = find(mask(:));
    if numel(setNodeIndex{iSet}) < 1
        toDelete(iSet) = true;
    else
        toDelete(iSet) = false;
    end
    setNodeWeight{iSet} = ones(length(setNodeIndex{iSet}), 1);
end

setNodeIndex(toDelete) = [];
setNodeWeight(toDelete) = [];
constrSetSize = length(setNodeIndex);

setWeight = nan(constrSetSize, 1);
setSum = nan(constrSetSize, 1);
for iSet = 1 : constrSetSize
    setSum(iSet) = sum(gtImage(setNodeIndex{iSet}) .* setNodeWeight{iSet} );
    setWeight(iSet) = 1.0; 
end

[maxRowColumnLoss, badY] = getMaxSetHammingLoss( gtImage, 1.0, setNodeIndex, setNodeWeight, setSum, setWeight, objSeed, bkgSeed);
lossWeight = 1.0 / maxRowColumnLoss;

% aggregate sets to compute the loss
loss = 0;
for iSet = 1 : constrSetSize
    loss = loss + setWeight(iSet) * abs( sum(resultImage(setNodeIndex{iSet}) .* setNodeWeight{iSet}) - setSum(iSet) );
end
loss = loss * lossWeight;

end
