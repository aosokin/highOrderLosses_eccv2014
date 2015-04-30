function [W, Xi, trainingInfo] = train_sSVM_nSlack(X_input, Y, options)
%train_sSVM_nSlack implements the n-slack cutting plane algorithm for training of the SSVM 
% Input:
%       X_input - cell array of objects; should be a structure with fields:
%               dataFile - mat-file for to load feature on the fly    
%               unaryFeatures - unary features per (super)-pixel: double(numNodes, unaryFeatureNum) 
%               pairwiseFeatures - pairwise features per edge: double(numEdges + 2, pairwiseFeatureNum)(1-st, 2-nd columns - indices of incident nodes, 3-rd - last - features)
%               nodeMap - a map of all nodes: double[imageHeight, imageWidth]: shows the ordering of pixels and their grouping into superpixels
%
%       Y - cell array of correct labels;
%       options - structure with fields:
%           classNum - number of classes of each node
%           unaryFeatureNum - number of unary features
%           pairwiseFeatureNum - number of pairwise features (only Potts models are now supported)
%           C - C parameter of sSVM (default: 1)
%           psi - pointer to the function to compute the generalized features
%           oracle - pointer to the oracle function
%           QP_method - choose QP optimization library
%               'dual_matlab' (default) - built'in quadprog for the dual
%               'primal_matlab' - built'in quadprog for the primal
%               'dual_mosek' - solving the dual by Mosek's quadprog
%               'primal_mosek' - solving the primal by Mosek's quadprog
%                 Note: Mosek is several times faster than matlab, to use it rename it's quadprog and optimset to mskquadprog and mskoptimset
%           eps - eps for violating constraints (default: 1e-2)
%           maxIter - maximum number of iterations (default: 100)
%           maxOracleCall - maximum number of oracle calls (default: inf)
%           maxTime - maximum time for training (default : inf)
%           oraclePerQP - number of oracle calls between quadprog calls (defaults: inf)
%           epsInactive - threshold to admit constraint inactive (default: 1)
%           constrDropThreshhold - after this number of iterations of being inactive during the quadprog the constraint is dropped (default: 10)
%           distW - maximum possible value of loss function per image (default: length(Y{1}))
%           badConstraintEps - threshhold to detect constraints whose slack is greater than the current oracle (default: 1e-1)
%           badConstraintWayout - shows what to do with bad constraints
%               0 - (default) - do nothing; good for undergenerating sSVM
%               1 - get rid of them after the detection
%               2 - lower their constant to match the new slack
%           negativePairwiseWeights - flag for keeping all pairwise weights non-positive (default: false)
%           loadDataInMemory - flag for loading all the data in the memory (default: true)
%           initW - initial values of weights W (default: zeros)
%           initXi - initial values of slacks Xi (default: zeros)
%           initWorkingSet - initial working set (default: empty)
%           initXiNum - index of the slack variable in each inequality of workingSet
%           trainingHistory - load histoty of training info
%           tempFile - temp file to solve progress on each iteration (default: 'SSVM_lastIteration.mat');
%           iterPerSave - how often to save the current status (default: every 10 iterations)
%
% Output:
%       W - vector with learned weights
%       Xi - vector with slack variables Xi
%       trainingInfo - optimization data: structure with fields:
%           fValue - the final functional value
%           workingSet - linear inequalities on W that constrain the working set
%           xiNum      - number of slack var xi that corresponds to each constraint
%           fValuePlot - plot of QP solver results
%           slackPlot  - plot of slack values
%           regPlot - plot of regularizer values
%           fValuePlot2 - plot of functional values after oracles;
%               meaningfull if options.oraclePerQP = inf;
%           constraintNumberPlot - number of present constraints at each QP-call
%
%   Anton Osokin (firstname.lastname@gmail.com)


%% Initialization
% initialize 
global X_dataset
X_dataset = X_input;

% init method
[config, W, Xi, workingSet, xiNum] = parseInput(X_input, Y, options);
Xi = zeros(size(Xi));
% features on training objects
fprintf('Computing features on the training objects.\n');
trainFeatures = zeros(config.N, config.M);
for iObject = 1 : config.N
    fprintf('*')
    trainFeatures(iObject, :) = config.psi(iObject, Y{iObject}, config)';
end
fprintf('\n')

inactiveConstraints = zeros(size(xiNum, 1), 1);
% init plots
[fValuePlot, fValuePlot2, regPlot, slackPlot, constraintNumberPlot, timePlot, W_history] = initPlots(config.trainingHistory);

% init time
tStart = tic;
if numel(timePlot) == 0
    oldHistoryTime = 0;
else
    oldHistoryTime = max(timePlot);
end

%% main loop
oracleCallNumber = 0;
iteration = 1;
% backup parameters for adaptive normalization
config.initDistW = config.distW;
config.initC = config.C;
while iteration <= config.maxIter
    fprintf('Iteration %d:\n', iteration);
    changed = false(config.N, 1);
    
    % loop to work with group of objects
    startObject = 1;
    addWorkingSet = cell(config.N, 1);
    addXiNum = cell(config.N, 1);
    
    if any(W ~= 0)
        normalizationMultiplyer = config.weightNorm / norm(W);
    else
        normalizationMultiplyer = 1;
    end
    
    
    config.epsNormalization = normalizationMultiplyer;
    
    while startObject <= config.N
        % run oracles possibly in parallel
        endObject = min(startObject + config.oraclePerQP - 1, config.N);
        tOracle = tic;
        for iObject = startObject : endObject
            %% find most violating constraint
            config.distW = config.initDistW * normalizationMultiplyer; 
            [F, fC, warmStart] = config.oracle(iObject, Y{iObject}, W * normalizationMultiplyer, config);
            
            config.distW = config.initDistW; 
            fC = fC / normalizationMultiplyer;
            
            % update the oracle counter
            oracleCallNumber = oracleCallNumber + 1;
            
            if ~(any(isnan(F(:))) || any(isnan(fC(:))))
                curXi = fC - (trainFeatures(iObject, :) - F') * W;
                
                % save warmStart for the next run
                X_dataset{iObject}.warmStart = warmStart;
                
                % if violation is large enough
                if curXi > Xi(iObject) + config.eps / normalizationMultiplyer;
                    fprintf('+');
                    Xi(iObject) = curXi;
                    addWorkingSet{iObject} = [trainFeatures(iObject, :) - F', -fC];
                    addXiNum{iObject} = iObject;
                    changed(iObject) = true;
                elseif curXi < Xi(iObject) - config.badConstraintEps / normalizationMultiplyer
                    % bad constraint detected
                    fprintf('-');
                    if config.badConstraintWayout == 1
                        % drop bad constraints
                        oldXi = -workingSet * [W; 1];
                        toDrop = (oldXi - config.badConstraintEps / normalizationMultiplyer > curXi) & (xiNum == iObject);
                        
                        workingSet(toDrop, :) = [];
                        xiNum(toDrop) = [];
                        inactiveConstraints(toDrop) = [];
                        
                        % add new constraint
                        Xi(iObject) = curXi;
                        addWorkingSet{iObject} = [trainFeatures(iObject, :) - F', -fC];
                        addXiNum{iObject} = iObject;
                        changed(iObject) = true;
                        
                    elseif config.badConstraintWayout == 2
                        % update bad constraints
                        oldXi = -workingSet * [W; 1];
                        toDrop = (oldXi - config.badConstraintEps / normalizationMultiplyer > curXi) & (xiNum == iObject);
                        residual = oldXi - config.badConstraintEps / normalizationMultiplyer - curXi;
                        residual(~toDrop) = [];
                        
                        workingSet(toDrop, end) = workingSet(toDrop, end) + residual;
                        
                        % add new constraint
                        Xi(iObject) = curXi;
                        addWorkingSet{iObject} = [trainFeatures(iObject, :) - F', -fC];
                        addXiNum{iObject} = iObject;
                        changed(iObject) = true;
                        
                    end
                    
                else
                    Xi(iObject) = curXi;
                    fprintf('.');
                end
            else
                X_dataset{iObject}.warmStart = [];
                fprintf('!');
            end
        end
        fprintf('\n');
        fprintf('Oracle time: %f\n', toc(tOracle));
        
        
        % update working Set
        workingSet = [workingSet; cat(1, addWorkingSet{startObject : endObject})];
        xiNum = [xiNum; cat(1, addXiNum{startObject : endObject})];
        inactiveConstraints = [inactiveConstraints; zeros(length(xiNum) - length(inactiveConstraints), 1)];
        
        %% solve QP program
        if any(changed(startObject:endObject))
            fValuePlot2(end + 1) = 0.5 * sum(W .^ 2, 1) / config.C + sum(Xi, 1) / config.N ;
            
            config.C = config.initC * normalizationMultiplyer;
            workingSetBackUp = workingSet(:, end);
            workingSet(:, end) = workingSetBackUp * normalizationMultiplyer;
            [W, Xi, fValue, info] = config.QP(workingSet, xiNum, config);
            config.C = config.initC;
            workingSet(:, end) = workingSetBackUp;
            W = W / normalizationMultiplyer;
            Xi = Xi / normalizationMultiplyer;
            fValue = fValue / normalizationMultiplyer;
            
            % edit warm start flag
            if info.exitflag ~= 1
                inactiveConstraints = 0 * inactiveConstraints;
            else
                % if everything is ok update info about inactive
                % constraints
                
                if size(workingSet, 1) > 0
                    inactiveConstraints(~info.inactiveConstr) = 0;
                    inactiveConstraints = inactiveConstraints + info.inactiveConstr;
                    
                    toDrop = (inactiveConstraints > config.constrDropThreshhold);
                    if any(toDrop)
                        workingSet(toDrop, :) = [];
                        xiNum(toDrop) = [];
                        inactiveConstraints(toDrop) = [];
                    end
                end
            end
            
            
            % update plots
            fValuePlot(end + 1) = fValue;
            slackPlot(end + 1) = sum(Xi, 1) / config.N;
            regPlot(end + 1) = 0.5 * sum(W .^ 2, 1) / config.C;
            constraintNumberPlot(end + 1) = length(inactiveConstraints);
            timePlot(end + 1) = toc(tStart) + oldHistoryTime;
            W_history{end + 1} = W;
        end
        
        % exit on maximum oracle calls
        if oracleCallNumber >= config.maxOracleCall
            break;
        end
        curTime = toc(tStart);
        if curTime >= config.maxTime
            break;
        end
        
        startObject = endObject + 1;
    end
    
    
    %% print current position
    fprintf('Average slack: %f, working set size: %d, weight norm: %f\n', mean(Xi), size(workingSet, 1), norm(W));
    
    % save data about the current iteration
    if mod(iteration, config.iterPerSave) == 0
        save(config.tempFile);
    end
    
    iteration = iteration + 1;
    
    if (~any(changed))
        fprintf('Working set is not changing.\n');
        break;
    end;
    
    if oracleCallNumber >= config.maxOracleCall
        warning('sSVM:termination', 'Maximum number of oracle calls exceeded. sSVM did not converge.');
        break;
    end
    
    if curTime >= config.maxTime
        warning('sSVM:termination', 'Maximum time reached. sSVM did not converge.');
        break;
    end
    
end

if iteration > config.maxIter
    warning('sSVM:termination', 'Maximum number of iterations exceeded. sSVM did not converge.');
end

%% compute obtained functional value
curXi = nan(config.N, 1);
for iObject = 1 : config.N
    % find the worst configuration
    [F, fC] = config.oracle(iObject, Y{iObject}, W, config);
    % compute hinge-loss
    curXi(iObject) = fC - (trainFeatures(iObject, :) - F') * W;
end
fValue = 0.5 * sum(W .^ 2, 1) / config.C + sum(curXi, 1) / config.N;

%% make trainingInfo output
trainingInfo = struct;
trainingInfo.workingSet = workingSet;
trainingInfo.xiNum = xiNum;
trainingInfo.fValuePlot = fValuePlot;
trainingInfo.fValuePlot2 = fValuePlot2;
trainingInfo.regPlot = regPlot;
trainingInfo.slackPlot = slackPlot;
trainingInfo.constraintNumberPlot = constraintNumberPlot;
trainingInfo.fValue = fValue;
trainingInfo.timePlot = timePlot;
trainingInfo.W_history = W_history;

%% clear global variables
clear global X_dataset
end

%% QP solvers
function [W, Xi, fValue, info] = solveQP_primal(workingSet, xiNum, config)
% info - structure with data about optimization; fields:
%           exitflag:   1 - everything is OK
%                       0 - QP solver failed to converge - no warm start
%                       -1- QP solver crashed

try
    %% Get parameters
    N = config.N; %number of objects
    M = config.M; %number of features
    C = config.C; % sSVM parameter
    
    Xi0 = zeros(config.N, 1);
    W0 = zeros(config.M, 1);
    
    %% Create problem structure
    Prob = struct;
    
    % Target function
    Prob.H = diag([ones(M, 1); zeros(N, 1)]);
    Prob.f = [zeros(M, 1); C *ones(N, 1)/ N];
    
    % Inequalities
    if size(workingSet, 1) > 0
        tmp = eye(N);
        Prob.Aineq = double([-workingSet(:, 1 : M), -tmp(xiNum, :)]);
        Prob.bineq = double(workingSet(:, M + 1));
    else
        Prob.Aineq = zeros(1, M + N);
        Prob.bineq = 0;
    end
    
    % Bound constraints
    Prob.lb = [-inf(M, 1); zeros(N, 1)];
    tmp = inf(M, 1);
    if config.negativePairwiseWeights
        %set predefined negative sign for some weights (positive from energy the point of view)
        tmp(end - config.pairwiseFeatureNum + 1 : end) = 0;
    end
    Prob.ub = [tmp; inf(N, 1)];
    
    % Starting point
    Prob.x0 = [W0; Xi0];
    
    %% Run QP solver
    
    [x, fValue, exitflag] = config.QP_solver(Prob);
    % Extract variables
    W = x(1 : M);
    Xi = x(M + 1 : M + N);
    
    % recompute Xi
    ineqXi = -workingSet * [W; 1];
    realXi = accumarray(xiNum, ineqXi, [N 1] , @max, -inf);
    Xi = max(realXi, 0);
    
    fValue = fValue / C;
    
    % Check exitflag
    info = struct;
    if exitflag == 1
        info.exitflag = 1;
    else
        info.exitflag = 0;
    end
    
    %% Check constraints
    info.inactiveConstr = (Prob.Aineq * [W; Xi] - Prob.bineq < -config.epsInactive);
    
catch err
    warning('sSVM:QP_primal', ['QP crashed: ', err.message ,'\n Try to continue']);
    
    W = zeros(config.M, 1);
    [fValue, Xi] = computeValue(-workingSet(:, 1 : M)', -workingSet(:, M + 1), config.C, config.N, xiNum, W);
    
    info = struct;
    info.exitflag = -1;
    info.inactiveConstr = false;
    return;
end

end

function [W, Xi, fValue, info] = solveQP_dual(workingSet, xiNum, config)
% info - structure with data about optimization; fields:
%           exitflag:   1 - everything is OK
%                       0 - QP solver failed to converge - no warm start
%                       -1- QP solver crashed

try
    
    %% Get parameters
    N = config.N; %number of objects
    M = config.M; %number of features
    C = config.C; % sSVM parameter
    K = size(workingSet, 1); % number of inequalities
    
    %% Create problem structure
    Prob = struct;
    lambda = N / C;
    if config.negativePairwiseWeights
        % additional dual variables to deal with negative weights
        gammaNum = config.pairwiseFeatureNum;
    else
        gammaNum = 0;
    end
    
    A = -workingSet(:, 1 : M)';
    b = -workingSet(:, M + 1);
    
    gammaAlphaMatrix = -workingSet(:, M - gammaNum + 1 : 1 : M)';
    % Target function
    Prob.H = [(A' * A), gammaAlphaMatrix';  gammaAlphaMatrix, eye(gammaNum)];
    Prob.f = [-b; zeros(gammaNum, 1)] * lambda;
    
    
    % Inequalities
    id = eye(N);
    Prob.Aineq = [id(:, xiNum), zeros(N, gammaNum)];
    Prob.bineq = ones(N, 1);
    
    % Bound constraints
    Prob.lb = zeros(K + gammaNum, 1);
    Prob.ub = ones(K + gammaNum, 1);
    
    % Regulirize the matrix
    Prob.H = Prob.H + eye(length(Prob.H)) * 1e-12;
    
    %% Run QP solver
    [x, f, exitflag] = config.QP_solver(Prob);
    f = f / lambda;
    
    % Extract variables
    W = (-1/lambda) * A * x(1 : K);
    W(end - gammaNum + 1 : 1 : end) = W(end - gammaNum + 1 : 1 : end) - x(end - gammaNum + 1 : 1 : end) / lambda;
    
    [fValue, Xi] = computeValue(A, b, C, N, xiNum, W);
    
    if abs(f / N + fValue) > 1e-2
%         warning('sSVM:QP_dual', 'Gap between the primal and the dual detected');
        fprintf('WARNING: Gap between the primal and the dual detected: %f\n', abs(f / N + fValue))
    end
    
    % Check exitflag
    info = struct;
    if exitflag == 1
        info.exitflag = 1;
    else
        info.exitflag = 0;
    end
    
    %% Check constraints
    
    info.inactiveConstr = (A' * W + b < Xi(xiNum) - config.epsInactive);
    
catch err
    warning('sSVM:QP_dual', ['QP crashed: ', err.message ,'\n Try to continue']);
    
    W = zeros(config.M, 1);
    [fValue, Xi] = computeValue(-workingSet(:, 1 : M)', -workingSet(:, M + 1), config.C, config.N, xiNum, W);
    
    info = struct;
    info.exitflag = -1;
    info.inactiveConstr = false;
    return;
end
end

function [f, xi] = computeValue(A, b, C, N, xiNum, W)
tmp = A' * W + b;
xi = accumarray(xiNum, tmp, [N 1], @max, -inf);
xi = max(xi, 0);
f = (1 / C / 2) * (W' * W) + sum(xi) / N;
end

function [x, fValue, exitflag] = solveQP_MATLAB(Prob)
%solve sSVM quadratic program
fprintf('Starting QUADPROG QP solver: ');
tStart = tic;

options = optimset('Algorithm', 'interior-point-convex', 'Display', 'off', 'TolFun', 1e-10, 'TolX', 1e-10, 'TolCon', 1e-10);

% start QP solver;
[x, fValue, exitflag] = quadprog(Prob.H + eye(size(Prob.H, 1)) * (1e-15), Prob.f, Prob.Aineq, Prob.bineq, [], [], Prob.lb, Prob.ub, [], options);
if exitflag < 0
    save('error_quadprog.mat', 'Prob', 'options');
    warning('sSVM:QUADPROG', ['QUADPROG error: exit flag: ', num2str(exitflag)]);
end
if exitflag == 0
    warning('sSVM:QUADPROG', 'QUADPROG: The maximum number of iterations was reached.');
end

tQP = toc(tStart);
fprintf('solution time: %f\n', tQP);
end

function [x, fValue, exitflag] = solveQP_MOSEK(Prob)
% solve sSVM quadratic program
fprintf('Starting MOSEK QP solver: ');
tStart = tic;

options = mskoptimset('Diagnostics', 'off', ' Display', 'off');

%start QP solver;
[x, fValue, exitflag] = quadprog(Prob.H + eye(size(Prob.H, 1)) * (1e-20), Prob.f, Prob.Aineq, Prob.bineq, [], [], Prob.lb, Prob.ub, [], options);
if exitflag < 0
    save('error_mosek.mat', 'Prob', 'options');
    error('sSVM:MOSEK', ['MOSEK error: exit flag: ', num2str(exitflag)]);
end
if exitflag == 0
    warning('sSVM:MOSEK', 'MOSEK: The maximum number of iterations was reached.');
end

tQP = toc(tStart);
fprintf('solution time: %f\n', tQP);
end


%% parse the input
function [config, W, Xi, workingSet, xiNum] = parseInput(X, Y, options)
% the function that parses input data
config = struct;

% number of objects
config.N = length(X);

% number of classes
config.K = options.classNum;

% number of unary features
config.unaryFeatureNum = options.unaryFeatureNum;

% number of pairwise features
config.pairwiseFeatureNum = options.pairwiseFeatureNum;

% number of weights
if options.classNum == 2
    config.M = options.unaryFeatureNum  + options.pairwiseFeatureNum; % this is a binary problem
else
    config.M = options.unaryFeatureNum * options.classNum + options.pairwiseFeatureNum;
end

% C - parameter
if ~isfield(options, 'C')
    options.C = 1;
end
config.C = options.C;

% choose oracle type
if ~isfield(options, 'oracle')
    error('options.oracle not specified')
end
config.oracle = options.oracle;

% choose function to compute feature vector
if ~isfield(options, 'psi')
    error('options.psi not specified')
end
config.psi = options.psi;

% QP optimization method
if ~isfield(options, 'QP_method')
    options.QP_method = 'dual_matlab';
end
switch lower(options.QP_method)
    case 'dual_matlab'
        config.QP = @solveQP_dual;
        config.QP_solver = @solveQP_MATLAB;
    case 'primal_matlab'
        config.QP = @solveQP_primal;
        config.QP_solver = @solveQP_MATLAB;
    case 'dual_mosek'
        config.QP = @solveQP_dual;
        config.QP_solver = @solveQP_MOSEK;
    case 'primal_mosek'
        config.QP = @solveQP_primal;
        config.QP_solver = @solveQP_MOSEK;
    otherwise
        error('Unknown QP optimization method!');
end
config.QP_method = options.QP_method;

% eps for violating constraints
if ~isfield(options, 'eps')
    options.eps = 1e-2;
end
config.eps = options.eps;

% eps for identifying bad constraints
if ~isfield(options, 'badConstraintEps')
    options.badConstraintEps = 1e-1;
end
config.badConstraintEps = options.badConstraintEps;

% what to do with bad constraints
if ~isfield(options, 'badConstraintWayout')
    options.badConstraintWayout = 0;
end
if ~ismember(options.badConstraintWayout, [0, 1, 2])
    warning('sSVM:parseInit', 'options.badConstraintWayout has a bad value - reset to the default');
    options.badConstraintWayout = 0;
end
config.badConstraintWayout = options.badConstraintWayout;

% distance weight
if ~isfield(options, 'distW')
    options.distW = max(length(Y{1}), 1);
end
config.distW = options.distW;
if isfield(options, 'distWeightAlpha')
    config.distWeightAlpha = options.distWeightAlpha;
end

% flag if pairwise weights in the energy habe to be kept positive
if ~isfield(options, 'negativePairwiseWeights')
    options.negativePairwiseWeights = false;
end
config.negativePairwiseWeights = options.negativePairwiseWeights;

% flag for loading all the data in the memory
if ~isfield(options, 'loadDataInMemory')
    options.loadDataInMemory = false;
end
config.loadDataInMemory = options.loadDataInMemory;



% maximum number of iterations
if ~isfield(options, 'maxIter')
    options.maxIter = 100;
end
config.maxIter = options.maxIter;

% maximum number of oracle calls
if ~isfield(options, 'maxOracleCall')
    options.maxOracleCall = inf;
end
config.maxOracleCall = options.maxOracleCall;

% maximum time for the method
if ~isfield(options, 'maxTime')
    options.maxTime = inf;
end
config.maxTime = options.maxTime;

% frequency of QP solver calls
if ~isfield(options, 'oraclePerQP')
    options.oraclePerQP = inf;
end
config.oraclePerQP = min(options.oraclePerQP, config.N);
config.oraclePerQP = round(max(options.oraclePerQP, 1));

% threshold to admit constraint inactive
if ~isfield(options, 'epsInactive')
    options.epsInactive = 1e-2;
end
config.epsInactive = options.epsInactive;

% threshold to admit constraint inactive
if ~isfield(options, 'constrDropThreshhold')
    options.constrDropThreshhold = 10;
end
config.constrDropThreshhold = options.constrDropThreshhold;

if ~isfield(options, 'iterPerSave')
    options.iterPerSave = 10;
end
config.iterPerSave = options.iterPerSave;

if ~isfield(options, 'weightNorm')
    options.weightNorm = 1e+2;
end
config.weightNorm = options.weightNorm;


%% defaults
% working set
workingSet = zeros(0, config.M + 1);
xiNum = zeros(0, 1);
% slack variables
Xi = zeros(config.N, 1);
% weights
W = zeros(config.M, 1);

if isfield(options, 'initW')
    if ~(any(size(W) ~= size(options.initW)))
        W = options.initW;
    else
        warning('sSVM:parseInit', 'options.initW is of wrong size, therefore ignored');
    end
end

if isfield(options, 'initXi')
    if ~any(size(Xi) ~= size(options.initXi))
        Xi = options.initXi;
    else
        warning('sSVM:parseInit', 'options.initXi is of wrong size, therefore ignored');
    end
end

if isfield(options, 'initWorkingSet') || isfield(options, 'initXiNum')
    if isfield(options, 'initWorkingSet') && isfield(options, 'initXiNum')
        if (size(workingSet, 2) == size(options.initWorkingSet, 2)) && (size(xiNum, 2) == size(options.initXiNum, 2)) && (size(options.initXiNum, 1) == size(options.initWorkingSet, 1)) && isempty(setdiff(unique(options.initXiNum), 1 : config.N ))
            workingSet = options.initWorkingSet;
            xiNum = options.initXiNum;
        else
            warning('sSVM:parseInit', 'options.initWorkingSet and/or options.initXiNum contain errors, therefore ignored');
        end
    else
        warning('sSVM:parseInit', 'options.initWorkingSet and options.initXiNum both should be specified to be used');
    end
end

if isfield(options, 'trainingHistory')
    config.trainingHistory = options.trainingHistory;
else
    config.trainingHistory = [];
end

% temporary file name
if ~isfield(options, 'tempFile')
    options.tempFile = 'SSVM_lastIteration.mat';
end
config.tempFile = options.tempFile;

end

function [fValuePlot, fValuePlot2, regPlot, slackPlot, constraintNumberPlot, timePlot, W_history] = initPlots(trainingHistory)

if isempty(trainingHistory)
    fValuePlot = zeros(0);
    fValuePlot2 = zeros(0);
    regPlot = zeros(0);
    slackPlot = zeros(0);
    constraintNumberPlot = zeros(0);
    timePlot = zeros(0);
    W_history = cell(0, 1);
else
    fValuePlot = trainingHistory.fValuePlot;
    fValuePlot2 = trainingHistory.fValuePlot2;
    regPlot = trainingHistory.regPlot;
    slackPlot = trainingHistory.slackPlot;
    constraintNumberPlot = trainingHistory.constraintNumberPlot;
    timePlot = trainingHistory.timePlot;
    W_history = trainingHistory.W_history;
end

end

