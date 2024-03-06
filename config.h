// config.h
#ifndef CONFIG_H_
#define CONFIG_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include <limits.h>
#include <float.h>
#include <omp.h>

#include <sys/stat.h>
#include <sys/types.h>

#define DEBUG 1 
#define WIDTH 20
#define HEIGHT 20
#define BIAS 20.0
#define SAMPLE_SIZE 75
#define TRAIN_PASSES 2000

#define PPM_SCALER 25
#define PPM_COLOR_INTENSITY 255
#define PPM_RANGE 10.0

#define DATA_FOLDER "data"

#define TRAIN_SEED 69
#define CHECK_SEED 420

typedef float Layer[HEIGHT][WIDTH];

// Function declarations
void layer_save_as_csv(Layer layer, const char *file_path);
void layer_load_from_csv(Layer weights, const char *file_path);
void layer_fill_rect(Layer layer, int x, int y, int w, int h, float value);
void layer_fill_circle(Layer layer, int cx, int cy, int r, float value);
void layer_save_as_ppm(Layer layer, const char *file_path);
void layer_save_as_bin(Layer layer, const char *file_path);
void layer_load_from_bin(Layer layer, const char *file_path);
void add_inputs_from_weights(Layer inputs, Layer weights);
void sub_inputs_from_weights(Layer inputs, Layer weights);
int rand_range(int low, int high);
void layer_random_rect(Layer layer);
void layer_random_circle(Layer layer);
float feed_forward(Layer inputs, Layer hidden_weights, Layer output_weights);
int train_pass(Layer inputs, Layer hidden_weights, Layer output_weights);
int check_pass(Layer inputs, Layer hidden_weights, Layer output_weights);

#endif // CONFIG_H_

