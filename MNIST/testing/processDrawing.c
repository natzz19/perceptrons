#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <stdbool.h>
#include "MNIST_MLP.h"
#include "MNIST_image.h"


int main (int argc, char **argv) {
    if (argc < 2) {
            printf("Usage: %s <softmax.weigths>\n", argv[0]);
            return 1;
        }

    // index.html sends to server and server forwards it to our binary
    float drawing[PIXELS];
    for (int i = 0; i < PIXELS; i++) {
        if (scanf("%f", &drawing[i]) == -1) return 1;
    }
    Image image = {.pixels = drawing};

    FILE *weights_fd = fopen(argv[1], "rb");
    CHECK_NULL(weights_fd, "weights_fd in main");

    fseek(weights_fd, 0, SEEK_END);
    size_t param_amt = ftell(weights_fd) / sizeof(float);
    fseek(weights_fd, 0, SEEK_SET);

    float *params = malloc(sizeof(float) * param_amt);
    fread(params, sizeof(float), param_amt, weights_fd);

    Net net = {
        .image = &image,
        .params = params,
    };
    forward(&net, true);

    // print all probs to stdout, server.py receives it
    for (int p = 0; p < OUTPUT_NEURONS; p++) {
        printf("%f ", net.o[p]);
    }
    puts("");

    return 0;
}
