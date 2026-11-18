#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include "MNIST_MLP.h"
#include "MNIST_image.h"



int main(int argc, char **argv) {

    if (argc < 3) {
            printf("Usage: %s <path/to/images> <path/to/labels>\n", argv[0]);
            return 1;
        }

    srand(time(NULL));

    size_t image_amt;
    Image **images = loadDataset(argv[1], argv[2], &image_amt);
    Image *imagePool = images[0]; // initImages() performs pool-allocation. images gets shuffled so we track the idx.
    float params[PARAMS];
    initParams(params);


    int epochs = 0;
    float epochLoss = 0;
    while (epochs < MAX_EPOCHS) {

        for (int b = 0; b < image_amt; b += BATCH_SIZE) {
            int count = (image_amt - b < BATCH_SIZE) ? image_amt - b : BATCH_SIZE; // if SETS is not divisible by BATCH_SIZE
            epochLoss += trainBatch((const Image **)&images[b], count, params);
        }
        epochs++;

        epochLoss /= image_amt;
        fprintf(stderr, "%f\n", epochLoss);
        if (epochLoss < LOSS_TARGET) break;
        epochLoss = 0;

        shuffle(images, image_amt);
    }
    printf("Finished Training in %d epochs\n", epochs);

    FILE *weights = fopen("out/mnist_mlp_128_softmax.CE.weights", "wb");
    fwrite(params, sizeof(float), PARAMS, weights);
    fclose(weights);

    freeData(images, imagePool);

    return 0;
}
