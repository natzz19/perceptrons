#ifndef MNIST_MLP_H
#define MNIST_MLP_H

#include <stdlib.h>
#include <stdbool.h>
#include "MNIST_image.h"

#define BATCH_SIZE 32
#define INPUTS PIXELS
#define HIDDEN_NEURONS 128
#define OUTPUT_NEURONS 10

#define HIDDEN_STRIDE (INPUTS + 1)
#define OUTPUT_STRIDE (HIDDEN_NEURONS + 1)

/* Each hidden neuron has 2 weights and 1 bias while each output layer HIDDEN_NEURONS ammount of weights and 1 bias */
#define OUTPUT_INDEX (HIDDEN_NEURONS * HIDDEN_STRIDE)
#define PARAMS (HIDDEN_NEURONS*HIDDEN_STRIDE + OUTPUT_STRIDE*OUTPUT_NEURONS)

#define PARAMSOF(arr, l, i) \
    ((arr) + ((l) == 'H' ? ((i) * HIDDEN_STRIDE) : (OUTPUT_INDEX + ((i) * OUTPUT_STRIDE))))


#define LR 0.04f
#define MAX_EPOCHS 1000000
#define LOSS_TARGET 0.035f


typedef struct {
    const Image *image;
    float *params;
    float h[HIDDEN_NEURONS];
    float o[OUTPUT_NEURONS];
    float grads[PARAMS];

    float loss;
    size_t argmax;
} Net;


float forward(Net *net, bool is_softmax);
float trainBatch(const Image **images, int count, float *params);
void initParams(float *params);



#endif
