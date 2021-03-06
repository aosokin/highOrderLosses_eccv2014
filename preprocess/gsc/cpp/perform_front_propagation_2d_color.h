#ifndef _PERFORM_FRONT_PROPAGATION_2D_H_
#define _PERFORM_FRONT_PROPAGATION_2D_H_

#include <math.h>
#include "config.h"
#include <stdio.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include "mex.h"

// some global variables
extern int n;			// width
extern int p;			// height
extern int nCh; // Number of channels in the image
extern int nPixels;
extern double* D;
extern double* S;
extern double* W;
extern double* Q;
extern double* L;
extern double* start_points;
extern double* end_points;
extern double* H;
extern int nb_iter_max;
extern int nb_start_points;
extern int nb_end_points;

typedef bool (*T_callback_intert_node)(int i, int j, int ii, int jj);

// main function
void perform_front_propagation_2d(T_callback_intert_node callback_insert_node = NULL);

#endif // _PERFORM_FRONT_PROPAGATION_2D_H_
