function mask = readBinaryImage( imageName )

gtImage = im2double(imread(imageName));
if numel(size(gtImage)) == 3
    gtImage = rgb2gray(gtImage);
end

mask = nan(size(gtImage, 1), size(gtImage, 2));
mask(gtImage < 0.1) = 0;
mask(gtImage > 0.9) = 1;

end

