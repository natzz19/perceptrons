#ifndef MNIST_IMAGE_H
#define MNIST_IMAGE_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define IMAGE_HEADER 0x10
#define LABEL_HEADER 0x8
#define WIDTH 28
#define HEIGHT 28
#define PIXELS (WIDTH * HEIGHT)

#define CHECK_NULL(ptr, func) do { \
    if (!(ptr)) { \
        perror(func); \
        exit(1); \
    } \
} while (0)

typedef struct {
    float *pixels;
    uint8_t label;
} Image;


Image **initImages(const uint8_t *pixels, const uint8_t *labels, size_t image_amt);
Image **loadDataset(const char *images_path, const char *labels_path, size_t *out_count);
void freeData(Image **images, Image *pool);
void shuffle(Image **images, size_t n);
void printLabel(const Image *image, FILE *fd);

#endif // MNIST_IMAGE_H
