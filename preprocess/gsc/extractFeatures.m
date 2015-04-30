function [features, edges, edge_weights, edge_type, colorWeights] = extractFeatures(img, opts)

if(~strcmp(class(img),'double')),
  error('img should be of type double in preProcessing\n');
end

[h,w,nCh]=size(img);

switch(opts.posteriorMethod)
  case {'gmm_bs','gmm_bs_mixtured'}
    switch(opts.featureSpace)
      case 'rgb'
        features=extractPixels(img);
      case 'rgb_xy'
        features=extractPixels(img);
        features=vag_addSpatialFeatures(features,img,10);
      otherwise
        error('Invalid feature space %s\n',opts.featureSpace);
    end
  otherwise
    error('Invalid posterior method %s\n',opts.posteriorMethod);
end

N = h*w; % the number of nodes
gamma=opts.gcGamma;

switch(opts.gcNbrType)
  case {'colourGradient','colourGradient8','colourGradient24',}
    switch(opts.gcNbrType)
      case 'colourGradient'
        roffset = [ 1,  0 ]; % 4 nbrhood
        coffset = [  0, 1  ];
      case 'colourGradient8'
        roffset = [ 1, 1, 0, -1 ];
        coffset = [  0, 1, 1, 1 ];
      case 'colourGradient24' % AOSOKIN
        roffset = [  1, 0, 1, -1, 0, 2,  2, -2, -2, -1, 1, 2  ];
        coffset = [  0, 1, 1,  1, 2, 0, -2,  2,  1,  2, 2, 1  ];
    end

    [lEdges,rEdges,colorWeights,spWeights]=...
    mex_setupTransductionGraph(img,int32(roffset'),int32(coffset'));

    if(isnumeric(opts.gcSigma_c)),
      beta=1/(2*D*opts.gcSigma_c^2);
    elseif(strcmp(opts.gcSigma_c,'auto'))
      beta=1/(0.5*mean(colorWeights));
      %fprintf('Auto beta in pp =%.4f\n',beta);
    else
      error('Improper sigma_c in transduction preprocessing\n');
    end

    edgeWeights = exp(-beta*colorWeights);
    %clear colorWeights spWeights;
    edgeWeights=gamma*edgeWeights;

    edgeWeights=opts.gcScale*edgeWeights;

    edge_type = spWeights(:)';

    %W=sparse(lEdges,rEdges,edgeWeights,N,N);
    %clear edgeWeights lEdges rEdges;
    %W=W+W';
    %whosW=whos('W');
    %if(whosW.bytes>N*N*8)
    %  fprintf('Its cheaper to store W as a full matrix rather than sparse!! converting\n');
    %  W=full(W);
    %end

  otherwise
    error('Invalid edge type \n');
end;

edge_weights = edgeWeights;
edges = [lEdges(:)'; rEdges(:)'];
