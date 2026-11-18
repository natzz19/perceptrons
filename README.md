# perceptrons

Neural networks from scratch in plain C, no libraries beyond the standard math library.

I built this mostly to demystify neural networks. I wanted to see for myself what a multi-layer perceptron actually is, and to get hands-on with backpropagation and gradient descent instead of just hearing the words. It turns out to be mostly calculus, some linear algebra, and a little probability (softmax/weight initialization). No ML libraries; so the forward pass and the gradients are written out explicitly.

## Contents

| Folder | What it is |
| --- | --- |
| `SLP/` | A single-layer perceptron learning an AND gate (can be switched to any linear problems like OR). |
| `xor/` | A small multi-layer perceptron (2-4-1, sigmoid) that solves XOR, with backprop written out step by step. |
| `MNIST/` | A 784-128-10 network (ReLU hidden layer, softmax output, cross-entropy loss) that recognises handwritten digits. It reaches about 97.79% on the MNIST test set. |
| `MNIST/testing/` | A tool that tests saved weights on the test set, plus a small web page where you can draw a digit and see the network's guess live. |

## Build

```
make          # build everything into builds/
make mnist    # or just one target: mnist, test, testDraw, slp, xor
```

## Run

The simple ones:

```
./builds/SLP
./builds/xorMLP
```

### MNIST

The dataset isn't included. Download and gunzip the four MNIST files and put them in `MNIST/assets/` named `TrainingImages`, `TrainingLabels`, `TestImages` and `TestLabels`.
Here is a link: https://systemds.apache.org/datasets/mnist

Run these from inside `MNIST/`, since the paths are relative:

```
cd MNIST

# train (writes out/mnist_mlp_128_softmax.CE.weights when it finishes)
../builds/MNIST_main assets/TrainingImages assets/TrainingLabels

# test saved weights
../builds/MNIST_test_weights assets/TestImages assets/TestLabels out/mnist_mlp_128_softmax.CE.weights
```

Training prints the loss per epoch to stderr. Weights are only saved when training ends, so let it finish.

### Draw a digit

After training, from the project root:

```
make serve
```

Then open http://localhost:8000 and draw. The page does the same preprocessing as MNIST (crop, scale to 20x20, center by center of mass) and sends the pixels to the C binary, which answers with the probabilities.

## A note on how this was made

The neural network code (the C in `SLP/`, `xor/` and `MNIST/`) was written by hand, as the point was to learn how it works. Becuase of that, its not robust and probably have bugs. I used an LLM (Claude) for the parts that aren't about the learning: the drawing page (`MNIST/testing/index.html`) and the small web server (`MNIST/testing/server.py`).
