function mask = findDifficultImages(dataset)
%findDifficultImages reads the list of 'difficult' images from file difficultImageList.txt

difficultImageFile = 'difficultImageList.txt';

diffImageNames = cell(0, 1);

% read the list form a file
fileID = fopen(difficultImageFile, 'r');
curLine = fgetl(fileID);
while length(curLine) > 2
    diffImageNames{end + 1} = curLine;
    curLine = fgetl(fileID);
end
fclose(fileID);

mask = false(length(dataset), 1);
for iImage = 1 : length(dataset)
   for iSample = 1 : length( diffImageNames )
       if ~isempty(strfind(dataset{iImage}, ['\', diffImageNames{iSample}, '.mat'])) || ...
           ~isempty(strfind(dataset{iImage}, ['/', diffImageNames{iSample}, '.mat'])) 
           mask(iImage) = true;
           break;
       end
   end
end

end
