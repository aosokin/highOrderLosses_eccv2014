function setup
%setup adds all the paths required by this package

rootDir = fileparts( mfilename( 'fullpath' ) );

addpath( fullfile(rootDir, 'graphCutMex_IBFS') );
addpath( fullfile(rootDir, 'preprocess') );
addpath( fullfile(rootDir, 'preprocess', 'gsc') );
addpath( fullfile(rootDir, 'data' ) );
addpath( fullfile(rootDir, 'training' ) );
addpath( fullfile(rootDir, 'training', 'losses' ) );
addpath( fullfile(rootDir, 'training', 'oracles' ) );
addpath( fullfile(rootDir, 'training', 'utils' ) );

end
