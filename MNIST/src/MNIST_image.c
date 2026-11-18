#include "MNIST_image.h"
#include <stdio.h>
#include <stdlib.h>

Image **initImages(const uint8_t *pixels, const uint8_t *labels, size_t image_amt) {

    Image **images = malloc(sizeof(Image *) * image_amt);
    CHECK_NULL(images, "Images in initData");

    Image *pool = malloc(sizeof(Image) * image_amt);
    CHECK_NULL(pool, "pool in initData");
    for (int img = 0; img < image_amt; img++) {
        images[img] = &pool[img];
    }

    float *pixelsNormalized = malloc(PIXELS * image_amt * sizeof(float));
    CHECK_NULL(pixelsNormalized, "pixelsNormalized in initData");
    for (int p = 0; p < PIXELS*image_amt; p++) {
        pixelsNormalized[p] = ((float)pixels[p]) * (1.0f / 255.0f);
    }
    for (int i = 0; i < image_amt; i++) {
        images[i]->pixels = &pixelsNormalized[PIXELS*i];
        images[i]->label  = labels[i];
	if (images[i]->label > 9) {
	    fprintf(stderr, "In initImages(): Loaded label OOB: %d\n", images[i]->label);
	    exit(1);
	}
    }

    return images;
}

Image **loadDataset(const char *images_path, const char *labels_path, size_t *out_count) {
    FILE *images_fd = fopen(images_path, "rb");
    FILE *labels_fd = fopen(labels_path, "rb");
    CHECK_NULL(images_fd, "images_fd in loadDataset");
    CHECK_NULL(labels_fd, "labels_fd in loadDataset");

    fseek(images_fd, 0, SEEK_END);
    size_t image_amt = (ftell(images_fd) - IMAGE_HEADER) / PIXELS;

    fseek(images_fd, IMAGE_HEADER, SEEK_SET);
    fseek(labels_fd, LABEL_HEADER, SEEK_SET);

    uint8_t *pixels = malloc(image_amt * PIXELS);
    uint8_t *labels = malloc(image_amt);
    CHECK_NULL(pixels, "pixels in loadDataset");
    CHECK_NULL(labels, "labels in loadDataset");

    fread(pixels, PIXELS, image_amt, images_fd);
    fread(labels, 1, image_amt, labels_fd);
    fclose(images_fd);
    fclose(labels_fd);

    Image **images = initImages(pixels, labels, image_amt);
    free(pixels);
    free(labels);

    if (out_count) {
        *out_count = image_amt;
    }
    return images;
}


void freeData(Image **images, Image *imagePool) {
    if (!images) return;
    free(imagePool->pixels);
    free(imagePool);
    free(images);
}

void shuffle(Image **images, size_t n) {
    if (n == 0) return;
    for (size_t i = n-1; i > 0; i--) {
        size_t j = (size_t)((double)rand() / ((double)RAND_MAX+1) * (i + 1));
        Image *tmp = images[i];
        images[i] = images[j];
        images[j] = tmp;
    }
}

void printLabel(const Image *image, FILE *fd) {
    const float (*pixels)[WIDTH] = (const float (*)[WIDTH])image->pixels;

    for (int row = 0; row < HEIGHT; row++) {
        for (int col = 0; col < WIDTH; col++) {
            char p;
            float strength = pixels[row][col];
            if (strength < 0.2f)     p = ' ';
            else if (strength < 0.5) p = '.';
            else                     p = '#';

            fprintf(fd, "%c", p);

        }
        fprintf(fd, "\n");
    }

    fprintf(fd, "\n\n");

}
