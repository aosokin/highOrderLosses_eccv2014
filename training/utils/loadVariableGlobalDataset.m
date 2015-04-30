function [variables, loaded] = loadVariableGlobalDataset(variableNames, iObject, loadDataInDataset)
%loadVariableGlobalDataset loads data from global variables 'X_dataset' of from file
%   if a field in 'X_dataset' does not exist function tryes to load it from file X_dataset{iObject}.dataFile
%
% [variables, loaded] = loadVariableGlobalDataset(variableNames, iObject);
% [variables, loaded] = loadVariableGlobalDataset(variableNames, iObject, loadDataInDataset);
%
% INPUT
%   variablesNames - list of variables to load; cell array NumVar x 1
%   iObject - index of object in X_dataset for loading; integer 1 x 1
%   loadDataInDataset - flag for saving loaded data to X_dataset{iObject}; logical 1 x 1; default: true
%
% OUTPUT
%   variables - cell array with loaded variables; cell array NumVar x 1;
%   loaded - vector showing if the loading was successfull; logical NumVar x 1
%
%   Anton Osokin, 27.11.2012

%% analyze input parameters
if nargin < 2 || nargin > 3
    error([mfilename, ':badNumberOfInputs'], 'Wrong number of input parameters');
end
if nargout > 2
    error([mfilename, ':badNumberOfOutputs'], 'Wrong number of output parameters');
end

if ischar(variableNames)
    variableNames = {variableNames};
end
if ~iscell(variableNames)
    error([mfilename, ':badInput_variableNames'], 'Input <<variablesNames>> is of wrong type');
end
variableNames = variableNames(:);

if ~isnumeric(iObject) || length(iObject) ~= 1
    error([mfilename, ':badInput_iObject'], 'Input <<iObject>> is of wrong type');
end

if nargin == 2
    loadDataInDataset = true;
end
if ~islogical(loadDataInDataset) || length(loadDataInDataset) ~= 1
    error([mfilename, ':badInput_loadDataInDataset'], 'Input <<loadDataInDataset>> is of wrong type');
end

%% start the function
variableNumber = length(variableNames);
variables = cell(variableNumber, 1);
global X_dataset

if ~isfield(X_dataset{iObject}, 'dataFile')
    error([mfilename, ':dataFileNotSpecified'], ['X_dataset{',num2str(iObject), '} does not have field <<dataFile>>']);
end

% check which variables are in global dataset
toAdd = false(variableNumber, 1);
loaded = false(variableNumber, 1);
toAddVariableList = cell(0, 0);
for iVar = 1 : variableNumber
    if ~isfield(X_dataset{iObject}, variableNames{iVar});
        toAdd(iVar) = true;
        toAddVariableList{end + 1} = variableNames{iVar};
    end
end

% read the varibles from file
if any(toAdd)
    % check if the variables exist in a file
    fileVarList = whos('-file', X_dataset{iObject}.dataFile);
    fileVarList = cat(1, {fileVarList(:).name});
    toDelete = false( length(toAddVariableList), 1 );
    for iVar = 1 : length( toAddVariableList )
        TF = strcmp(toAddVariableList{iVar}, fileVarList);
        if ~any(TF)
            toDelete(iVar) = true;
            % error([mfilename, ':nonExistentVariable'],  ['Variable ',variableList{iVar}, ' does not exist in file for object #', num2str(iObject), ' : ', X_dataset{iObject}.dataFile]);
        end
    end
    toAddVariableList( toDelete ) = [];
    if ~isempty( toAddVariableList )
        % read the variables form a file
        newVars = load(X_dataset{iObject}.dataFile, toAddVariableList{:});
    else
        newVars = struct;
    end
end

% add new variables to the output and, if necessary, to the global data structure
for iVar = 1 : variableNumber
    if toAdd(iVar)
        if isfield( newVars, variableNames{iVar} )
            variables{iVar} = newVars.(variableNames{iVar});
            if loadDataInDataset
                X_dataset{iObject}.(variableNames{iVar}) = variables{iVar};
            end
            loaded(iVar) = true;
        else
            loaded(iVar) = false;
        end
    else
        if isfield( X_dataset{iObject}, variableNames{iVar} )
            variables{iVar} = X_dataset{iObject}.(variableNames{iVar});
            loaded(iVar) = true;
        else
            loaded(iVar) = false;
        end
    end
end

end
