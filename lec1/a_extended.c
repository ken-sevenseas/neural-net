/*************************************************************/
/* C-program for delta-learning rule                         */
/* Learning rule of one neuron                               */
/*                                                           */
/* This program is produced by Qiangfu Zhao.                 */
/* You are free to use it for educational purpose            */
/*************************************************************/
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define I 3
#define n_sample 4
#define eta 0.5
#define lambda 1.0
#define desired_error 0.01
#define sigmoid(x) (2.0 / (1.0 + exp(-lambda * x)) - 1.0)
#define frand() (rand() % 10000 / 10001.0)
#define randomize() srand((unsigned int)time(NULL))

//(0,0,-1); (0,1,-1); (1,0,-1); (1,1,-1)
double x[n_sample][I] = {
    {0, 0, -1},
    {0, 1, -1},
    {1, 0, -1},
    {1, 1, -1},
};

double w[I];
double d[n_sample] = {-1, -1, -1, 1};
double o;

void deltaLearning(void);
void perceptronLearning(void);
void Initialization(void);
void FindOutput(int);
void FindOutputPerceptron(int);
void PrintResult(void);
void printNeuronOutput();

int main() {
    deltaLearning();
    perceptronLearning();
    return 0;
}

/*************************************************************/
/* implement of each learning rule                           */
/*************************************************************/
void deltaLearning() {
    int i, p, q = 0;
    double delta, Error = DBL_MAX;

    Initialization();
    printf("The initial connection weights of the neurons:\n");
    printNeuronOutput();
    printf("\n\n");

    while (Error > desired_error) {
        q++;
        Error = 0;
        for (p = 0; p < n_sample; p++) {
            FindOutput(p);
            Error += 0.5 * pow(d[p] - o, 2.0);
            for (i = 0; i < I; i++) {
                delta = (d[p] - o) * (1 - o * o) / 2;
                w[i] += eta * delta * x[p][i];
            }
            printf("Error in the %d-th learning cycle=%f\n", q, Error);
        }
    }
    PrintResult();
    printNeuronOutput();
}

void perceptronLearning() {
    int i, p, q = 0;
    double LearningSignal = 1.0, Error = DBL_MAX;

    Initialization();
    printf("The initial connection weights of the neurons:\n");
    printNeuronOutput();
    printf("\n\n");

    while (Error > desired_error) {
        q++;
        Error = 0;
        for (p = 0; p < n_sample; p++) {
            FindOutputPerceptron(p);
            Error += 0.5 * pow(d[p] - o, 2.0);
            LearningSignal = eta * (d[p] - o);
            for (i = 0; i < I; i++) {
                w[i] += LearningSignal * x[p][i];
            }
        }
        printf("Error in the %d-th learning cycle=%f\n", q, Error);
    }
    PrintResult();
    printNeuronOutput();
}

/*************************************************************/
/* Initialization of the connection weights                  */
/*************************************************************/
void Initialization(void) {
    int i;

    randomize();
    for (i = 0; i < I; i++) w[i] = frand();
}

/*************************************************************/
/* Find the actual outputs of the network                    */
/*************************************************************/
void FindOutput(int p) {
    int i;
    double temp = 0;

    for (i = 0; i < I; i++) temp += w[i] * x[p][i];
    o = sigmoid(temp);
}

void FindOutputPerceptron(int p) {
    int i;
    double temp = 0;

    for (i = 0; i < I; i++) temp += w[i] * x[p][i];
    o = temp > 0 ? 1 : -1;
}

/*************************************************************/
/* Print out the final result                                */
/*************************************************************/
void PrintResult(void) {
    int i;

    printf("\n\n");
    printf("The connection weights of the neurons:\n");
    for (i = 0; i < I; i++) printf("%5f ", w[i]);
    printf("\n\n");
}

void printNeuronOutput() {
    int p, i;
    double u;
    int output;

    printf("\nNeuron output for each input pattern:\n");
    for (p = 0; p < n_sample; p++) {
        u = 0.0;
        for (i = 0; i < I; i++) {
            u += w[i] * x[p][i];
        }
        output = (u > 0) ? 1 : -1;
        printf("Input (%f, %f, %f) -> Output: %d (Teacher: %f)\n", x[p][0],
               x[p][1], x[p][2], output, d[p]);
    }
}