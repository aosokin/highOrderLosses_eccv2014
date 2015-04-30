function addVariableGlobalDataset(variableName, iObject, data)
%addVariableGlobalDataset adds data to the global variables 'X_dataset' 
% 
% addVariableGlobalDataset(variableName, iObject, data)
% 
% INPUT:
%   variableName - a field name to add to X_dataset{iObject}; char array 1 x length(variableName)  
%   iObject - index of object in X_dataset for loading; integer 1 x 1
%   data - data to save to field X_dataset{iObject}.variableName
%
%   Anton Osokin, 27.11.2012

%% analyze input parameters
if nargin ~= 3
    error([mfilename, ': Wrong number of input parameters']);
end
if nargout > 0
    error([mfilename, ': Wrong number of output parameters']);
end

if ischar(variableName)
    variableName = {variableName};
end
if ~iscell(variableName)
    error([mfilename, ': variableName of wrong type']);
end
variableName = variableName(:);

if ~isnumeric(iObject) || length(iObject) ~= 1
    error([mfilename, 'addVariableGlobalDataset: iObject of wrong type']);
end

%% start the fucntion

global X_dataset
for iVar = 1 : length(variableName)
    X_dataset{iObject}.(variableName{iVar}) = data;
end

end