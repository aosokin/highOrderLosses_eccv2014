function compile_mex
%compile_mex compiles all the MEX-functions included in this package
%
% Anton Osokin

rootDir = fileparts( mfilename( 'fullpath' ) );

% IBFS GraphCut
if exist( 'graphCutMex', 'file' ) ~= 3
    cd( fullfile(rootDir, 'graphCutMex_IBFS') );
    build_graphCutMex;
    cd(rootDir);
end 

% compile the GSC code
cd( fullfile(rootDir, 'preprocess', 'gsc') );
compile_mex_gsc;
cd(rootDir);

end
