// Major edits (largely undocumented) made by Rushang Karia.
// This is for the 4ops version of blocksworld.
// gcc blocksworld.c bwstates.c -lm -w -o blocksworld

// DISCLAIMER: Only basic testing done.

/*********************************************************************
 * (C) Copyright 2001 Albert Ludwigs University Freiburg
 *     Institute of Computer Science
 *
 * All rights reserved. Use of this software is permitted for 
 * non-commercial research purposes, and it may be copied only 
 * for that use.  All copies must include this copyright message.
 * This software is made available AS IS, and neither the authors
 * nor the  Albert Ludwigs University Freiburg make any warranty
 * about the software or its performance. 
 *********************************************************************/


#include <stdlib.h>
#include <stdio.h>
#include <strings.h>
#include <sys/timeb.h>
#include <math.h>
#include "bwstates.h"

#define MAX_LENGTH 256
#define MAX_SAMPLE 50

/* data structures ... (ha ha)
 */
typedef unsigned char Bool;
#define TRUE 1
#define FALSE 0



int main( int argc, char *argv[] )
{

    // Need S >= 0 for get_options() to work correctly.
    int S = 0;            /* RK: [NOP] Number of states required */
    state sigma;          /* Structure holding the state */
    float *ratio;         /* Array of ratios. See note in BW.h */
    long seed;            /* Seed for drand48() */
    int x;
    struct timeb tp;
    ftime( &tp );

    ratio = (float*)malloc(sizeof(bigarray));
    sigma = (state)malloc(sizeof(STATE));
    sigma->N = 0;                                   /* Default */

    seed = (long)tp.millitm;                              /* Default */
    get_options(argc,argv,&(sigma->N),&S,&seed);    /* Read command line */
    make_ratio(sigma->N,ratio);                     /* Get probabilities */
    srand48(seed);                                  /* Initialise drand48() */


	make_state(sigma,ratio);

    state goal_sigma;
    goal_sigma = (state)malloc(sizeof(STATE));
    goal_sigma->N = sigma->N;
    float *goal_ratio = (float*)malloc(sizeof(bigarray));
    make_ratio(goal_sigma->N,goal_ratio);                     /* Get probabilities */
	make_state(goal_sigma, goal_ratio);


    FILE *data;

    int i, j;
    char c;

    int *initial, *goal;

    int gn = sigma->N;


    initial = ( int * ) calloc( gn + 1, sizeof( int ) );
    goal = ( int * ) calloc( gn + 1, sizeof( int ) );

    for ( i = 0; i < gn; i++ ) {

        initial[i + 1] = sigma->S[i] + 1;
    }



    for ( i = 0; i < gn; i++ ) {

        goal[i + 1] = goal_sigma->S[i] + 1;
    }


    printf("\n\n(define (problem BW-rand-%d)", gn);
    printf("\n(:domain blocksworld)");
    printf("\n(:objects ");
    for ( i = 0; i < gn; i++ ) printf("b%d ", i+1);
    printf(")");
    printf("\n(:init");
    printf("\n(arm-empty)");
    for ( i = 1; i < gn + 1; i++ ) {
    if ( initial[i] == 0 ) {
      printf("\n(on-table b%d)", i);
    } else {
      printf("\n(on b%d b%d)", i, initial[i]);
    }
    }
    for ( i = 1; i < gn + 1; i++ ) {
    for ( j = 1; j < gn + 1; j++ ) {
      if ( j == i ) continue;
      if ( initial[j] == i ) break;
    }
    if ( j < gn + 1 ) continue;
    printf("\n(clear b%d)", i);
    }
    printf("\n)");
    printf("\n(:goal");
    printf("\n(and");
    for ( i = 1; i < gn + 1; i++ ) {
    if ( goal[i] == 0 ) {

      printf("\n(on-table b%d)", i);
    } else {
      printf("\n(on b%d b%d)", i, goal[i]);
    }
    }
    printf(")\n)\n)\n\n\n");



    exit( 0 );

}

