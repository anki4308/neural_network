#include "config.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include <float.h>
#include <omp.h>

#include <sys/stat.h>
#include <sys/types.h>

#include <stdio.h>


#ifdef _clang_
#include <immintrin.h>
#elif defined(_GNUC_)
#include <x86intrin.h>
#endif


#ifndef DEBUG_PRINT
#define DEBUG_PRINT(fmt, ...) \
    do { \
        if (DEBUG) { \
            fprintf(stderr, fmt, ##__VA_ARGS__); \
        } \
    } while (0)
#endif



#define INPUT_SIZE WIDTH
#define HIDDEN_SIZE 64 
#define OUTPUT_SIZE 1

typedef float Layer[HEIGHT][WIDTH];

static inline int clampi(int x, int low, int high)
{
    if (x < low)  x = low;
    if (x > high) x = high;
    return x;
}




// Function to save weights to a CSV file
 void layer_save_as_csv(Layer layer, const char *file_path) {
    FILE *f = fopen(file_path, "w");
    if (f == NULL) {
        fprintf(stderr, "ERROR: could not open file %s: %s\n", file_path, strerror(errno));
        return;  // Return instead of exit(1) to allow program to continue
    }

    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            fprintf(f, "%f,", layer[y][x]);
        }
        fprintf(f, "\n");
    }

    fclose(f);
    printf("Weights saved to %s\n", file_path);
}

// Function to load weights from a CSV file
void layer_load_from_csv(Layer weights, const char *file_path)
{
    FILE *f = fopen(file_path, "r");
    if (f == NULL) {
        fprintf(stderr, "ERROR: could not open file %s: %s\n", file_path, strerror(errno));
        exit(1);
    }

    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            if (fscanf(f, "%f,", &weights[y][x]) != 1) {
                fprintf(stderr, "ERROR: invalid CSV format in file %s\n", file_path);
                exit(1);
            }
        }
    }

    fclose(f);
}

void layer_fill_rect(Layer layer, int x, int y, int w, int h, float value)
{
    assert(w > 0);
    assert(h > 0);
    int x0 = clampi(x, 0, WIDTH-1);
    int y0 = clampi(y, 0, HEIGHT-1);
    int x1 = clampi(x0 + w - 1, 0, WIDTH-1);
    int y1 = clampi(y0 + h - 1, 0, HEIGHT-1);

    #pragma omp parallel for
    for (int y = y0; y <= y1; ++y) {
        #pragma omp simd
        for (int x = x0; x <= x1; ++x) {
            layer[y][x] = value;
        }
    }
}

void layer_fill_circle(Layer layer, int cx, int cy, int r, float value)
{
    assert(r > 0);
    int x0 = clampi(cx - r, 0, WIDTH-1);
    int y0 = clampi(cy - r, 0, HEIGHT-1);
    int x1 = clampi(cx + r, 0, WIDTH-1);
    int y1 = clampi(cy + r, 0, HEIGHT-1);

    #pragma omp parallel for
    for (int y = y0; y <= y1; ++y) {
        #pragma omp simd
        for (int x = x0; x <= x1; ++x) {
            int dx = x - cx;
            int dy = y - cy;
            if (dx*dx + dy*dy <= r*r) {
                layer[y][x] = value;
            }
        }
    }
}

void layer_save_as_ppm(Layer layer, const char *file_path)
{
    FILE *f = fopen(file_path, "wb");
    if (f == NULL) {
        fprintf(stderr, "ERROR: could not open file %s: %s\n", file_path, strerror(errno));
        exit(1);
    }

    fprintf(f, "P6\n%d %d 255\n", WIDTH * PPM_SCALER, HEIGHT * PPM_SCALER);

    #pragma omp parallel for
    for (int y = 0; y < HEIGHT * PPM_SCALER; ++y) {
        #pragma omp simd
        for (int x = 0; x < WIDTH * PPM_SCALER; ++x) {
            float s = (layer[y / PPM_SCALER][x / PPM_SCALER] + PPM_RANGE) / (2.0f * PPM_RANGE);
            char pixel[3] = {
                (char) floorf(PPM_COLOR_INTENSITY * (1.0f - s)),
                (char) floorf(PPM_COLOR_INTENSITY * (1.0f - s)),
                (char) floorf(PPM_COLOR_INTENSITY * s),
            };

            fwrite(pixel, sizeof(pixel), 1, f);
        }
    }

    fclose(f);
}

void layer_save_as_bin(Layer layer, const char *file_path)
{
    FILE *f = fopen(file_path, "wb");
    if (f == NULL) {
        fprintf(stderr, "ERROR: could not open file %s: %s\n", file_path, strerror(errno));
        exit(1);
    }
    fwrite(layer, sizeof(Layer), 1, f);
    fclose(f);
}

void layer_load_from_bin(Layer layer, const char *file_path)
{
    FILE *f = fopen(file_path, "rb");
    if (f == NULL) {
        fprintf(stderr, "ERROR: could not open file %s: %s\n", file_path, strerror(errno));
        exit(1);
    }
    fread(layer, sizeof(Layer), 1, f);
    fclose(f);
}

void add_inputs_from_weights(Layer inputs, Layer weights)
{
    #pragma omp parallel for
    for (int y = 0; y < HEIGHT; ++y) {
        #pragma omp simd
        for (int x = 0; x < WIDTH; ++x) {
            weights[y][x] += inputs[y][x];
        }
    }
}

void sub_inputs_from_weights(Layer inputs, Layer weights)
{
    #pragma omp parallel for
    for (int y = 0; y < HEIGHT; ++y) {
        #pragma omp simd
        for (int x = 0; x < WIDTH; ++x) {
            weights[y][x] -= inputs[y][x];
        }
    }
}

int rand_range(int low, int high)
{
    assert(low < high);
    return rand() % (high - low) + low;
}

void layer_random_rect(Layer layer)
{
    layer_fill_rect(layer, 0, 0, WIDTH, HEIGHT, 0.0f);
    int x = rand_range(0, WIDTH);
    int y = rand_range(0, HEIGHT);

    int w = WIDTH - x;
    if (w < 2) w = 2;
    w = rand_range(1, w);

    int h = HEIGHT - x;
    if (h < 2) h = 2;
    h = rand_range(1, h);

    layer_fill_rect(layer, x, y, w, h, 1.0f);
}

void layer_random_circle(Layer layer)
{
    layer_fill_rect(layer, 0, 0, WIDTH, HEIGHT, 0.0f);
    int cx = rand_range(0, WIDTH);
    int cy = rand_range(0, HEIGHT);
    int r = INT_MAX;
    if (r > cx) r = cx;
    if (r > cy) r = cy;
    if (r > WIDTH - cx) r = WIDTH - cx;
    if (r > HEIGHT - cy) r = HEIGHT - cy;
    if (r < 2) r = 2;
    r = rand_range(1, r);
    layer_fill_circle(layer, cx, cy, r, 1.0f);
}

float feed_forward(Layer inputs, Layer hidden_weights, Layer output_weights)
{
    float hidden_output = 0.0f;

    #pragma omp parallel for reduction(+:hidden_output)
    for (int y = 0; y < HEIGHT; ++y) {
        #pragma omp simd
        for (int x = 0; x < WIDTH; ++x) {
            hidden_output += inputs[y][x] * hidden_weights[y][x];
        }
    }

    // a simple step function as the activation function for simplicity
    float hidden_activation = (hidden_output > BIAS) ? 1.0f : 0.0f;

    float final_output = 0.0f;

    #pragma omp parallel for reduction(+:final_output)
    for (int y = 0; y < HEIGHT; ++y) {
        #pragma omp simd
        for (int x = 0; x < WIDTH; ++x) {
            final_output += hidden_activation * output_weights[y][x];
        }
    }

    return final_output;
}

int train_pass(Layer inputs, Layer hidden_weights, Layer output_weights)
{

clock_t start_time = clock();

    static char file_path[256];
    static int count = 0;

    int adjusted = 0;

    #pragma omp parallel for reduction(+:adjusted)
    for (int i = 0; i < SAMPLE_SIZE; ++i) {
        layer_random_rect(inputs);
        if (feed_forward(inputs, hidden_weights, output_weights) > BIAS) {
            sub_inputs_from_weights(inputs, hidden_weights);
            snprintf(file_path, sizeof(file_path), DATA_FOLDER"/hidden_weights-%03d.ppm", count);
            DEBUG_PRINT("[INFO] saving %s\n", file_path);
            layer_save_as_ppm(hidden_weights, file_path);

            add_inputs_from_weights(inputs, output_weights);
            snprintf(file_path, sizeof(file_path), DATA_FOLDER"/output_weights-%03d.ppm", count++);
            DEBUG_PRINT("[INFO] saving %s\n", file_path);
            layer_save_as_ppm(output_weights, file_path);
            adjusted += 1;
        }

        layer_random_circle(inputs);
        if (feed_forward(inputs, hidden_weights, output_weights) < BIAS) {
            add_inputs_from_weights(inputs, hidden_weights);
            snprintf(file_path, sizeof(file_path), DATA_FOLDER"/hidden_weights-%03d.ppm", count);
            DEBUG_PRINT("[INFO] saving %s\n", file_path);
            layer_save_as_ppm(hidden_weights, file_path);

            sub_inputs_from_weights(inputs, output_weights);
            snprintf(file_path, sizeof(file_path), DATA_FOLDER"/output_weights-%03d.ppm", count++);
            DEBUG_PRINT("[INFO] saving %s\n", file_path);
            layer_save_as_ppm(output_weights, file_path);
            adjusted += 1;
        }
    }
 //// Stop benchmark
    clock_t end_time = clock();
    double duration = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
    printf("[BENCHMARK] Train Pass: %f seconds\n", duration);
    return adjusted;
}

int check_pass(Layer inputs, Layer hidden_weights, Layer output_weights)
{
clock_t start_time = clock();
    int adjusted = 0;

    #pragma omp parallel for reduction(+:adjusted)
    for (int i = 0; i < SAMPLE_SIZE; ++i) {
        layer_random_rect(inputs);
        if (feed_forward(inputs, hidden_weights, output_weights) > BIAS) {
            adjusted += 1;
        }

        layer_random_circle(inputs);
        if (feed_forward(inputs, hidden_weights, output_weights) < BIAS) {
            adjusted += 1;
        }
    }
 // Stop benchmark
    clock_t end_time = clock();
    double duration = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
    printf("[BENCHMARK] Check Pass: %f seconds\n", duration);
    return adjusted;
}


int main(void) {
    Layer hidden_weights;
    Layer output_weights;
    Layer inputs;

    printf("[INFO] creating %s\n", DATA_FOLDER);
    if (mkdir(DATA_FOLDER, 0755) < 0 && errno != EEXIST) {
        fprintf(stderr, "ERROR: could not create folder %s: %s", DATA_FOLDER, strerror(errno));
        exit(1);
    }

    // Load weights from CSV
    layer_load_from_csv(hidden_weights, "hidden_weights.csv");
    layer_load_from_csv(output_weights, "output_weights.csv");

    srand(CHECK_SEED);
    int adj = check_pass(inputs, hidden_weights, output_weights);
    printf("[INFO] fail rate of untrained model is %f\n", adj / (SAMPLE_SIZE * 2.0));

    for (int i = 0; i < TRAIN_PASSES; ++i) {
        // Re-initialize inputs with random values for each training pass
        srand(TRAIN_SEED);
        for (int y = 0; y < HEIGHT; ++y) {
            for (int x = 0; x < WIDTH; ++x) {
                inputs[y][x] = ((float)rand() / RAND_MAX) * 2.0 - 1.0; // Example random value generation between -1 and 1
            }
        }

        int adj = train_pass(inputs, hidden_weights, output_weights);
        printf("[INFO] Pass %d: adjusted %d times\n", i, adj);

        if (adj <= 0) {
            DEBUG_PRINT("[DEBUG] Exiting training loop as adjustments <= 0\n");
            break;
        }
    }

    srand(CHECK_SEED);
    adj = check_pass(inputs, hidden_weights, output_weights);
    printf("[INFO] fail rate of trained model is %f\n", adj / (SAMPLE_SIZE * 2.0));

 // Benchmark the train pass
    clock_t start_time_train = clock();
    int adj_train = train_pass(inputs, hidden_weights, output_weights);
    clock_t end_time_train = clock();
    double duration_train = ((double) (end_time_train - start_time_train)) / CLOCKS_PER_SEC;
    printf("[BENCHMARK] Total Train Pass: %f seconds, Adjusted: %d\n", duration_train, adj_train);

    // Benchmark the check pass
    clock_t start_time_check = clock();
    int adj_check = check_pass(inputs, hidden_weights, output_weights);
    clock_t end_time_check = clock();
    double duration_check = ((double) (end_time_check - start_time_check)) / CLOCKS_PER_SEC;
    printf("[BENCHMARK] Total Check Pass: %f seconds, Adjusted: %d\n", duration_check, adj_check);

    return 0;
}

