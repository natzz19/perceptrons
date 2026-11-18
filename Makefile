CC      = gcc
CFLAGS  = -O3 -march=native -ffast-math -Wall -Wextra -std=c11
LDFLAGS = -lm
BUILD   = builds

MNIST_SRCS = MNIST/src/MNIST_MLP.c MNIST/src/MNIST_image.c
MNIST_HDRS = $(wildcard MNIST/include/*.h)
MNIST_INC  = -IMNIST/include

# Binary names match what server.py expects (builds/MNIST_processDrawing).
TRAIN = $(BUILD)/MNIST_main
TEST  = $(BUILD)/MNIST_test_weights
DRAW  = $(BUILD)/MNIST_processDrawing
SLP   = $(BUILD)/SLP
XOR   = $(BUILD)/xorMLP

.PHONY: all clean mnist test testDraw slp xor serve

all: mnist test testDraw slp xor

mnist:    $(TRAIN)
test:     $(TEST)
testDraw: $(DRAW)
slp:      $(SLP)
xor:      $(XOR)

# order-only prerequisite (after |): create the dir without forcing rebuilds
$(BUILD):
	mkdir -p $@

$(TRAIN): MNIST/src/main.c $(MNIST_SRCS) $(MNIST_HDRS) | $(BUILD)
	$(CC) $(CFLAGS) MNIST/src/main.c $(MNIST_SRCS) $(MNIST_INC) -o $@ $(LDFLAGS)

$(TEST): MNIST/testing/test_weights.c $(MNIST_SRCS) $(MNIST_HDRS) | $(BUILD)
	$(CC) $(CFLAGS) MNIST/testing/test_weights.c $(MNIST_SRCS) $(MNIST_INC) -o $@ $(LDFLAGS)

$(DRAW): MNIST/testing/processDrawing.c $(MNIST_SRCS) $(MNIST_HDRS) | $(BUILD)
	$(CC) $(CFLAGS) MNIST/testing/processDrawing.c $(MNIST_SRCS) $(MNIST_INC) -o $@ $(LDFLAGS)

$(SLP): SLP/SLP.c | $(BUILD)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

$(XOR): xor/xorMLP.c | $(BUILD)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# server.py uses relative paths, so it must run from MNIST/testing
serve: $(DRAW)
	cd MNIST/testing && python3 server.py

clean:
	rm -rf $(BUILD)
