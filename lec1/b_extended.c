/*************************************************************/
/* C-program for learning of single layer neural network     */
/* based on the delta learning rule                          */
/*                                                           */
/*  1) Number of Inputs : N                                  */
/*  2) Number of Output : R                                  */
/* The last input for all neurons is always -1               */
/*                                                           */
/* This program is produced by Qiangfu Zhao.                 */
/* You are free to use it for educational purpose            */
/*************************************************************/
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 3
#define R 3
#define n_sample 3
#define eta 0.5
#define lambda 1.0
#define desired_error 0.1
#define sigmoid(x) (2.0 / (1.0 + exp(-lambda * x)) - 1.0)
#define frand() (rand() % 10000 / 10001.0)
#define randomize() srand((unsigned int)time(NULL))

double x[n_sample][N] = {
    {10, 2, -1},
    {2, -5, -1},
    {-5, 5, -1},
};
double d[n_sample][R] = {
    {1, -1, -1},
    {-1, 1, -1},
    {-1, -1, 1},
};
double w[R][N];
double o[R];

void deltaLearning();
void perceptronLearning(void);
void Initialization(void);
void FindOutput(int);
void FindOutputStep(int);
void PrintResult(void);
void printNeuronOutput(int);

int main() {
    int type;
    printf(
        "Choose the learning rule (1: discret (Perceptron learning), 2: "
        "continuous (Delta learning))\n");
    printf("Enter the type of learning rule (0 or 1): >>> ");
    scanf("%d", &type);
    if (type == 0) {
        deltaLearning();
    } else {
        perceptronLearning();
    }
    return 0;
}

/*************************************************************/
/* implement of each learning rule                           */
/*************************************************************/
void perceptronLearning() {
    int i, j, p, q = 0;
    double LearningSignal = 1.0, Error = DBL_MAX;

    Initialization();
    printf("The initial connection weights of the neurons:\n");
    printNeuronOutput(1);
    printf("\n\n");
    while (Error > desired_error) {
        q++;
        Error = 0;
        for (p = 0; p < n_sample; p++) {
            FindOutputStep(p);
            for (i = 0; i < R; i++) {
                Error += 0.5 * pow(d[p][i] - o[i], 2.0);
            }
            for (i = 0; i < R; i++) {
                LearningSignal = eta * (d[p][i] - o[i]);
                for (j = 0; j < N; j++) {
                    w[i][j] += LearningSignal * x[p][j];
                }
            }
        }
        printf("Error in the %d-th learning cycle=%f\n", q, Error);
    }
    PrintResult();
    printNeuronOutput(1);
}

void deltaLearning() {
    int i, j, p, q = 0;
    double Error = DBL_MAX;
    double delta;

    Initialization();
    printf("The initial connection weights of the neurons:\n");
    printNeuronOutput(2);
    printf("\n\n");
    while (Error > desired_error) {
        q++;
        Error = 0;
        for (p = 0; p < n_sample; p++) {
            FindOutput(p);
            for (i = 0; i < R; i++) {
                Error += 0.5 * pow(d[p][i] - o[i], 2.0);
            }
            for (i = 0; i < R; i++) {
                delta = (d[p][i] - o[i]) * (1 - o[i] * o[i]) / 2;
                for (j = 0; j < N; j++) {
                    w[i][j] += eta * delta * x[p][j];
                }
            }
        }
        printf("Error in the %d-th learning cycle=%f\n", q, Error);
    }
    PrintResult();
    printNeuronOutput(2);
    printf("\n\n");
}

/*************************************************************/
/* Initialization of the connection weights                  */
/*************************************************************/
void Initialization(void) {
    int i, j;

    randomize();
    for (i = 0; i < R; i++)
        for (j = 0; j < N; j++) w[i][j] = frand() - 0.5;
}

/*************************************************************/
/* Find the actual outputs of the network                    */
/*************************************************************/
void FindOutput(int p) {
    int i, j;
    double temp;

    for (i = 0; i < R; i++) {
        temp = 0;
        for (j = 0; j < N; j++) {
            temp += w[i][j] * x[p][j];
        }
        o[i] = sigmoid(temp);
    }
}
void FindOutputStep(int p) {
    int i, j;
    double temp;

    for (i = 0; i < R; i++) {
        temp = 0;
        for (j = 0; j < N; j++) {
            temp += w[i][j] * x[p][j];
        }
        o[i] = temp > 0 ? 1 : -1;
    }
}

/*************************************************************/
/* Print out the final result                                */
/*************************************************************/
void PrintResult(void) {
    int i, j;

    printf("\n\n");
    printf("The connection weights are:\n");
    for (i = 0; i < R; i++) {
        for (j = 0; j < N; j++) printf("%5f ", w[i][j]);
        printf("\n");
    }
    printf("\n\n");
}

void printNeuronOutput(int type) {
    // type = 1: perceptron learning, type = 2: delta learning;
    int p, i, j;
    double u, output;

    printf("\nNeuron output for each input pattern:\n");

    for (p = 0; p < n_sample; p++) {
        printf("Input (%f, %f, %f):\n", x[p][0], x[p][1], x[p][2]);

        for (i = 0; i < R; i++) {
            u = 0.0;
            for (j = 0; j < N; j++) {
                u += w[i][j] * x[p][j];
            }

            if (type == 1) {
                output = (u > 0) ? 1.0 : -1.0;
                printf("  Output[%d]: %.1f (Teacher: %.1f)\n", i, output,
                       d[p][i]);
            } else {
                output = sigmoid(u);
                printf("  Output[%d]: %f (Teacher: %.1f)\n", i, output,
                       d[p][i]);
            }
        }
        printf("\n");
    }
}
