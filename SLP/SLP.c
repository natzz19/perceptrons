#include <stdio.h>
#include <stdint.h>



int main(void) {
                      // x1  x2 target
    const int data1[3] = {0, 0, 0};
    const int data2[3] = {0, 1, 0};
    const int data3[3] = {1, 0, 0};
    const int data4[3] = {1, 1, 1};
    const int *matrice[4] = {data1, data2, data3, data4};

    float weights[2] = {0.0f, 0.0f};
    float bias = 0;
    const float lr = 1;

     // newWeight = weight + (lr * (target - prediciton) * x
     // prediction = 1 if score > 0 else 0
     // score = weight*x + bias

    int epochs = 0, sets = 4, correct = 0;
    while (correct != sets) {

        correct = 0;
        for (int i = 0; i < sets; i++) {

            int target = matrice[i][2];
            float score = 0;
            for (int j = 0; j < 2; j++) {
                score += matrice[i][j] * weights[j];
            }
            score += bias;
            int prediction = score > 0 ? 1 : 0;

            if (prediction == matrice[i][2]) correct++;

            for (int j = 0; j < 2; j++) {
                weights[j] += (lr * (target - prediction) * matrice[i][j]);
            }
            bias += (lr * (target - prediction));
        }

        epochs++;
    }

    printf("\nw1 = %f --- w2 = %f\nbias = %f\nepochs = %d\n", weights[0], weights[1], bias, epochs);

    return 0;
}

// sizeof(matrice) / sizeof(int*)
