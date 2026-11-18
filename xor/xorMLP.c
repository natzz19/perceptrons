#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


#define SETS 4
#define BATCH_SIZE 2
#define INPUTS 2
#define HIDDEN_NEURONS 4

#define HIDDEN_STRIDE (INPUTS + 1)
#define OUTPUT_STRIDE (HIDDEN_NEURONS + 1)

/* Each hidden neuron has 2 weights and 1 bias while each output layer HIDDEN_NEURONS ammount of weights and 1 bias */
#define OUTPUT_INDEX (HIDDEN_NEURONS * HIDDEN_STRIDE)
#define PARAMS (HIDDEN_NEURONS*HIDDEN_STRIDE + OUTPUT_STRIDE)

// l == layer, i == neuron idx
#define PARAMSOF(arr, l, i) \
    ((arr) + ((l) == 'H' ? ((i) * HIDDEN_STRIDE) : (OUTPUT_INDEX + ((i) * OUTPUT_STRIDE))))


#define LR 6.0f

#define MAX_EPOCHS 1000000
#define LOSS_TARGET 0.001f


typedef struct {
    float inputs[INPUTS];
    float target;
} Dataset;

typedef struct {
    const Dataset *data;
    float *params;
    float h[HIDDEN_NEURONS];
    float grads[PARAMS];

    float loss;
    float out;
} Net;


float calcActivation(const float *dataset, float *params, int n);
float forward(const float *dataset, Net *net);
float trainBatch(const Dataset *datasets, int count, float *params);
float getSlope(const Dataset *datasets, float *params, int n, float averageLossOld);
void backProp(Net *net);
float getSSE(const Dataset *datasets, float *params);

static inline float sigmoid(float n) {
    return 1 / (1 + expf(-n));
}

int main(void) {
    srand(time(NULL));

    const Dataset datasets[4] = {
        {.inputs = {0, 0}, .target = 0},
        {.inputs = {0, 1}, .target = 1},
        {.inputs = {1, 0}, .target = 1},
        {.inputs = {1, 1}, .target = 0},
    };

    float params[PARAMS] = {0};
    /* randomize each param [-1, 1] */
    for (int param = 0; param < PARAMS; param++) {
        params[param] = rand() / (float)RAND_MAX * 2.0f - 1.0f;
    }


    int epochs = 0;
    float batchLoss = 0, meanLoss = 0;
    while (epochs < MAX_EPOCHS) {

        for (int s = 0; s < SETS; s += BATCH_SIZE) {
            int count = (SETS - s < BATCH_SIZE) ? SETS - s : BATCH_SIZE; // if SETS is not divisible by BATCH_SIZE
            batchLoss = trainBatch(datasets + s, count, params);
        }
        epochs++;
        meanLoss = getSSE(datasets, params);
        if (meanLoss < LOSS_TARGET) break;
        if (epochs % 1000 == 0) {
            // printf("HN1:\nW1 = %f\tW2 = %f\n\n", params[0], params[1]);
            // printf("HN2:\nW1 = %f\tW2 = %f\n\n", params[3], params[4]);
            printf("Average batch loss = %f\n", batchLoss);
            printf("Average mean  loss = %f\n\n", meanLoss);
        }
    }

    for (int i = 0; i < SETS; i++) {
        Net net = {.params = params};
        float out = forward(datasets[i].inputs, &net);
        int result = (out > 0.5f) ? 1 : 0;
        printf("%d ^ %d == %d ->> %f\n", (int)datasets[i].inputs[0], (int)datasets[i].inputs[1], result, out);
    }
        puts("");

    return 0;
}

float trainBatch(const Dataset *datasets, int count, float *params) {
    Net net = {
        .params = params,
        .grads = {0},
    };

    float total_loss = 0.0f;
    for (int b = 0; b < count; b++) {

        net.data   = &datasets[b];
        net.out    = forward(net.data->inputs, &net);
        float diff = net.data->target - net.out;
        net.loss   = 0.5 * diff * diff; // loss = 1/2(target - output)²
        total_loss += net.loss;

        backProp(&net);
    }
    for (int p = 0; p < PARAMS; p++) {
        net.params[p] -= LR * (net.grads[p] / count);
    }

    return total_loss / count;
}

float getSSE(const Dataset *datasets, float *params) {

    float loss = 0.0f;
    for (int s = 0; s < SETS; s++) {
        float out = forward(datasets[s].inputs, &(Net){ .params = params });
        float diff = datasets[s].target - out;
        loss += 0.5 * diff * diff;
    }
    return loss / SETS;
}

float forward(const float *inputs, Net *net) {

    for (int i = 0; i < HIDDEN_NEURONS; i++) {
        net->h[i] = calcActivation(inputs, PARAMSOF(net->params, 'H', i), 2);
    }

    return calcActivation((const float *)net->h, PARAMSOF(net->params, 'O', 0), HIDDEN_NEURONS);
}

void backProp(Net *net) {

    // dLoss/dOut = (output - target)
    float delta = net->out - net->data->target;

    // dOut/dz = σ(z)(1 - σ(z))
    delta *= net->out*(1.0f - net->out);

    // dz/wx = hx
    for (int w = 0; w < HIDDEN_NEURONS; w++) {
        PARAMSOF(net->grads, 'O', 0)[w] += delta * net->h[w];
    }
    // dz/wb = 1
    PARAMSOF(net->grads, 'O', 0)[HIDDEN_NEURONS] += delta;


    for (int neuronBranch = 0; neuronBranch < HIDDEN_NEURONS; neuronBranch++) {
        float branchDelta = delta;

        // dz/hx = wx
        branchDelta *= PARAMSOF(net->params, 'O', 0)[neuronBranch];

        // dhx/dhxz --- derivative of this hidden-layers activation in respect to its z, score pre-sigmoid
        // σ(z)(1 - σ(z))
        branchDelta *= net->h[neuronBranch]*(1.0f - net->h[neuronBranch]);

        // dz/wx = ix
        for (int i = 0; i < INPUTS; i++) {
            PARAMSOF(net->grads, 'H', neuronBranch)[i] += branchDelta * net->data->inputs[i];
        }
        // dz/wb = 1
        PARAMSOF(net->grads, 'H', neuronBranch)[INPUTS] += branchDelta;
    }


}

float calcActivation(const float *inputs, float *params, int n) {

    float score = 0;
    for (int i = 0; i < n; i++) {
        score += (params[i] * inputs[i]);
    }
    score += params[n];


    return sigmoid(score);
}
