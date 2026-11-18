#include <math.h>
#include <stdbool.h>
#include "MNIST_MLP.h"


static inline float ReLU(float z) {
    return (z > 0.0f) ? z : 0.0f;
}

static inline size_t argmax(float *arr, int n) {
    int max = 0;
    for (int i = 1; i < n; i++) {
        if (arr[i] > arr[max]) max = i;
    }
    return max;
}

static inline void softmax(float *arr, int n) {

    // subtract higehest exponent from all exponent, yields same end ratio, and avoids NaN
    float max_z = arr[argmax(arr, n)];


    float sum = 0;
    for (int j = 0; j < n; j++) {
        arr[j] = expf(arr[j] - max_z);
        sum += arr[j];
    }

    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) {
        arr[i] *=  inv_sum;
    }

}

static inline float randUniform() {
    return ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f);
}

static inline float heXavier(int fan_in) {
    float dividend = (fan_in == HIDDEN_NEURONS) ? 6.0f : 3.0f;
    return sqrtf(dividend / (float)fan_in);
}


void initParams(float *params) {
    // 1. Hidden layer
    float hidden_factor = heXavier(INPUTS);
    for (int i = 0; i < OUTPUT_INDEX; i++) {
        if (i % HIDDEN_STRIDE == HIDDEN_STRIDE - 1) {
            params[i] = 0.0f;
        } else {
            params[i] = randUniform() * hidden_factor;
        }
    }

    // 2. Output layer
    float output_factor = heXavier(HIDDEN_NEURONS);
    for (int i = OUTPUT_INDEX; i < PARAMS; i++) {
        int local_idx = i - OUTPUT_INDEX;
        if (local_idx % OUTPUT_STRIDE == OUTPUT_STRIDE - 1) {
            params[i] = 0.0f;
        } else {
            params[i] = randUniform() * output_factor;
        }
    }
}

static float calcActivation(const float * restrict inputs, const float * restrict params, int n) {

    float score = 0;
    for (int i = 0; i < n; i++) {
        score += (inputs[i] * params[i]);
    }
    score += params[n];

    // no actvation if we are calcing output neurons
    return (n == HIDDEN_NEURONS) ? score : ReLU(score);
}

// when no activation on output used, linear raw logits
static float getSSE(const Net *net) {

    /*
               9
    // SSE = ½ Σ(target - output)²
              i=0
    */
    float loss = 0;
    for (int i = 0; i < OUTPUT_NEURONS; i++) {
        float target = (i == net->image->label) ? 1.0f : 0.0f;
        float diff = target - net->o[i];
        loss += diff * diff;
    }
    return 0.5 * loss;
}

// when using softmax on output activations
static float getCrossEntropy(const Net *net) {
    /*
               9
    // SSE = - Σ target[i] * ln(out[i])
              i=0
    */

    // becuase we use hot-labels, all other addends of the crossEntropy except cur_image->label becomes 0
    return -logf(net->o[net->image->label]+ 1e-7f);
}

static float getLoss(const Net *net, bool is_softmax) {
    return is_softmax ? getCrossEntropy(net) : getSSE(net);
}

float forward(Net *net, bool is_softmax) {

    for (int i = 0; i < HIDDEN_NEURONS; i++) {
        net->h[i] = calcActivation((const float * restrict)net->image->pixels, (const float * restrict)PARAMSOF(net->params, 'H', i), INPUTS);
    }

    for (int i = 0; i < OUTPUT_NEURONS; i++) {
        net->o[i] = calcActivation((const float * restrict)net->h, (const float * restrict)PARAMSOF(net->params, 'O', i), HIDDEN_NEURONS);
    }
    if (is_softmax) softmax(net->o, OUTPUT_NEURONS);



    net->argmax = argmax(net->o, OUTPUT_NEURONS);
    return getLoss(net, is_softmax);
}

static void backProp(Net *net) {

    float hiddenDeltas[HIDDEN_NEURONS] = {0};


    for (int outputBranch = 0; outputBranch < OUTPUT_NEURONS; outputBranch++) {


        // dΣLoss/dOut = (output - target) -> dOut/dz = 1 (no activation/linear)
        float target = (outputBranch == net->image->label) ? 1.0f : 0.0f;
        float delta = net->o[outputBranch] - target;


        // dz/wx = hx
        for (int w = 0; w < HIDDEN_NEURONS; w++) {
            PARAMSOF(net->grads, 'O', outputBranch)[w] += delta * net->h[w];
        }
        // dz/wb = 1
        PARAMSOF(net->grads, 'O', outputBranch)[HIDDEN_NEURONS] += delta;


        for (int hiddenBranch = 0; hiddenBranch < HIDDEN_NEURONS; hiddenBranch++) {
            float deltaHiddenBranch = delta;

            // dz/hx = wx
            deltaHiddenBranch *= PARAMSOF(net->params, 'O', outputBranch)[hiddenBranch];
            hiddenDeltas[hiddenBranch] += deltaHiddenBranch;
            // factor out the deltas for each hidden neuron acros all 10 outputs
            // (d1 + d2 + d.. + dn) * INPUTS instead of (d1*INPTUS) + (d2*INPUTS)...
        }
    }

    for (int hiddenBranch = 0; hiddenBranch < HIDDEN_NEURONS; hiddenBranch++) {
        float * restrict gradrow = PARAMSOF(net->grads, 'H', hiddenBranch);
        const float * restrict pixels = net->image->pixels;

        // dhx/dhxz --- derivative of this hidden-layers activation in respect to its z, score pre-ReLU
        // 1 if (hx > 0) else 0 -> if 0, following grads get no change
        hiddenDeltas[hiddenBranch] *= net->h[hiddenBranch] > 0 ? 1 : 0;
        float hd = hiddenDeltas[hiddenBranch];
        if (hd == 0) continue;

        // dz/wx = ix
        for (int i = 0; i < INPUTS; i++) {
            gradrow[i] += hd * pixels[i];
        }
        // dz/wb = 1
        gradrow[INPUTS] += hd;
    }
}


float trainBatch(const Image **images, int count, float *params) {
    Net net = {
        .params = params,
        .grads = {0},
        .h = {0},
    };

    float totalBatchLoss = 0.0f;
    for (int img = 0; img < count; img++) {

        net.image  = images[img];
        net.loss = forward(&net, true);
        totalBatchLoss += net.loss;

        backProp(&net);
    }

    float scale = LR / count;
    for (int p = 0; p < PARAMS; p++) {
        net.params[p] -= scale * net.grads[p];
    }

    return totalBatchLoss;
}
