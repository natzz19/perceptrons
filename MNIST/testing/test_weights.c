#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <stdbool.h>
#include "MNIST_MLP.h"
#include "MNIST_image.h"

#define LINEAR 1
#define SOFTMAX 2
#define BOTH true

int main (int argc, char **argv) {
    if (argc < 4) {
            printf("Usage: %s <images> <labels> <softmax.weigths> <linear.weigths>\n", argv[0]);
            return 1;
        }

    size_t image_amt;
    Image **images = loadDataset(argv[1], argv[2], &image_amt);

    // char *linear = "out/mnist_mlp_128_linear.SSE.weights";    // weights trained with linear output activations and SSE
    // char *softmax = "out/mnist_mlp_128_softmax.CE.weights";   // trained with softmax and crossEntropy

    int round = 0;
    run:

        FILE *weights_fd = fopen(argv[round+3], "rb");
        CHECK_NULL(weights_fd, "weights_fd in main");

        fseek(weights_fd, 0, SEEK_END);
        size_t param_amt = ftell(weights_fd) / sizeof(float);
        fseek(weights_fd, 0, SEEK_SET);

        float *params = malloc(sizeof(float) * param_amt);
        fread(params, sizeof(float), param_amt, weights_fd);

        bool is_softmax = round+3 == 3 ? true : false;

        size_t correct = 0;
        FILE *errors_fd = fopen(is_softmax ? "out/softmax_errors.txt" : "out/linear_errors.txt", "a");
        for (int img = 0; img < image_amt; img++) {
            Net net = {
                .image = images[img],
                .params = params,
            };

            forward(&net, is_softmax);

            if (net.argmax == images[img]->label) correct++;
            else {
                fprintf(errors_fd, "label: %d, predicted: %zu, ", images[img]->label, net.argmax);
                if (is_softmax) fprintf(errors_fd, "confidence: %f", net.o[net.argmax]);
                printLabel(images[img], errors_fd);
            }
        }
        float per = ((float)correct / (float)image_amt) * 100;
        printf("\n\nWeight-file:\n%s\nAmmount of test-images: %zu\nCorrect: %zu ---> %%%f\n\n", argv[round+3], image_amt, correct, per);

        round++;
    if (round+3 < argc) goto run;

    return 0;

}
