Perceptually Inspired Layout-aware Losses for Image Segmentation
================================================================


This software implements the method described in the following paper:

A. Osokin and P. Kohli
Perceptually Inspired Layout-aware Losses for Image Segmentation
In European Conference on Computer Vision (ECCV), 2014.

The paper can be downloaded from:
http://bayesgroup.ru/wp-content/uploads/2014/07/skeletalLossesLearning_eccv2014_cameraReady.pdf

If you use the software in your work please cite the aforementioned paper.




1. Installation
--------------------

1) Run setup.m to add the required folders to Matlab Path

2) Run compile_mex.m to compile all the MEX-functions on your system.
Tested with Matlab R2014a + MSVC 2012 on Win7 and Matlab R2014b + gcc 4.6 on Ubuntu  12.04

3) Download the dataset provided by the following paper:
V. Gulshan, C. Rother, A. Criminisi, A. Blake and A. Zisserman
Geodesic Star Convexity for Interactive Image Segmentation
In Computer Vision and Pattern Recognition (CVPR), 2010. 

Images: http://www.robots.ox.ac.uk/~vgg/data/iseg/data/images.tgz
Ground truth: http://www.robots.ox.ac.uk/~vgg/data/iseg/data/images-gt.tgz
Brush strokes: http://www.robots.ox.ac.uk/~vgg/data/iseg/data/images-labels.tgz

Unpack everything to ./data

4) Run ./experiments_eccv2014/example_training.m to test how everything is working
Run ./experiments_eccv2014/example_motivation.m to see Table 1 of the paper

5) Run ./experiments_eccv2014/run_full_experiment.m to reproduce the ECCV 2014 experiments
Note, the full experiment requires 280 runs of the training procedure which will take quite some time




2. Third-party software
--------------------

This software uses several third-party packages (some of the packages are released under the research-only license).
Please, consider citing the corresponding papers:

1) IBFS algorithm to solve the max-flow/min-cut problem:
http://www.cs.tau.ac.il/~sagihed/ibfs/

A. V. Goldberg, S. Hed, H. Kaplan, R. E. Tarjan, and R. F. Werneck, 
Maximum Flows by Incremental Breadth-First Search,
In Proceedings of the 19th European conference on Algorithms, ESA'11, pages 457-468.

2) Geodesic star convexity 
http://www.robots.ox.ac.uk/~vgg/software/iseg/

V. Gulshan, C. Rother, A. Criminisi, A. Blake and A. Zisserman,
Geodesic star convexity for interactive image segmentation. 
In Proceedings of Conference on Vision and Pattern Recognition (CVPR 2010).

3) Learning Low-order Models for Enforcing High-order Statistics
https://github.com/ppletscher/hol

P. Pletscher and P. Kohli
Learning Low-order Models for Enforcing High-order Statistics
AISTATS, 2012.
