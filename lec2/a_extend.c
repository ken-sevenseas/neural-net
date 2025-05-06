/*************************************************************/
/* C-program for BP algorithm                                */
/* The nerual network to be designed is supposed to have     */
/* three layers:                                             */
/*  1) Input layer : I inputs                                */
/*  2) Hidden layer: J neurons                               */
/*  3) Output layer: K neurons                               */
/* The last input is always -1, and the output of the last   */
/* hidden neuron is also -1.                                 */
/*                                                           */
/* This program is produced by Qiangfu Zhao.                 */
/* You are free to use it for educational purpose            */
/*************************************************************/
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define I 5          // 4 original inputs + 1 dummy input
#define J_MAX 10     // Maximum number of hidden neurons
#define K 1          // 1 output
#define n_sample 16  // 2^4 = 16 possible combinations for 4-bit input
#define eta 0.5
#define lambda 1.0
#define desired_error 0.001
#define sigmoid(x) (1.0 / (1.0 + exp(-lambda * x)))
#define frand() (rand() % 10000 / 10001.0)
#define randomize() srand((unsigned int)time(NULL))

// 実験タイプの定義を追加
#define PARITY_CHECK 0
#define XNOR 1

int J;  // Variable to change the number of hidden neurons
double x[n_sample][I];
double d[n_sample][K];
double v[J_MAX][I], w[K][J_MAX];
double y[J_MAX];
double o[K];

void Initialization(void);
void FindHidden(int p);
void FindOutput(void);
void PrintResult(void);
void GenerateTrainingDataParity(void);
void GenerateTrainingDataXNOR(void);
void RunExperimentParity(int hiddenNeurons);
void RunExperimentXNOR(int hiddenNeurons);

int main() {
    int choice;
    
    printf("Select experiment type:\n");
    printf("1. Parity check\n");
    printf("2. XNOR\n");
    printf("Enter choice (1 or 2): ");
    scanf("%d", &choice);
    
    if (choice == 1) {
        printf("\nRunning Parity Check experiments\n");
        // Test with different numbers of hidden neurons
        RunExperimentParity(4);
        RunExperimentParity(6);
        RunExperimentParity(8);
        RunExperimentParity(10);
    } else if (choice == 2) {
        printf("\nRunning XNOR experiments\n");
        // Test with different numbers of hidden neurons
        RunExperimentXNOR(4);
        RunExperimentXNOR(6);
        RunExperimentXNOR(8);
        RunExperimentXNOR(10);
    } else {
        printf("Invalid choice. Exiting.\n");
    }

    return 0;
}

// RunExperiment関数をRunExperimentParityに名前変更
void RunExperimentParity(int hiddenNeurons) {
    int i, j, k, p, q = 0;
    double Error = DBL_MAX;
    double delta_o[K];
    double delta_y[J_MAX];

    J = hiddenNeurons;
    printf("\n\n---------------------------------------------\n");
    printf("Running parity experiment with %d hidden neurons\n", J);
    printf("---------------------------------------------\n");

    GenerateTrainingDataParity();
    Initialization();

    while (Error > desired_error) {
        q++;
        Error = 0;
        for (p = 0; p < n_sample; p++) {
            FindHidden(p);
            FindOutput();

            for (k = 0; k < K; k++) {
                Error += 0.5 * pow(d[p][k] - o[k], 2.0);
                delta_o[k] = (d[p][k] - o[k]) * (1 - o[k]) * o[k];
            }

            for (j = 0; j < J; j++) {
                delta_y[j] = 0;
                for (k = 0; k < K; k++) delta_y[j] += delta_o[k] * w[k][j];
                delta_y[j] = (1 - y[j]) * y[j] * delta_y[j];
            }

            for (k = 0; k < K; k++)
                for (j = 0; j < J; j++) w[k][j] += eta * delta_o[k] * y[j];

            for (j = 0; j < J; j++)
                for (i = 0; i < I; i++) v[j][i] += eta * delta_y[j] * x[p][i];
        }

        if (q % 100 == 0 || Error <= desired_error) {
            printf("Error in the %d-th learning cycle = %f\n", q, Error);
        }

        // Avoid infinite loops
        // if (q > 10000) {
        //     printf("Max iterations reached. Stopping training.\n");
        //     break;
        // }
    }

    PrintResult();

    // Test the network with all samples
    printf("\nTest results for %d hidden neurons:\n", J);
    printf("Input\t\tDesired\tOutput\n");
    for (p = 0; p < n_sample; p++) {
        FindHidden(p);
        FindOutput();
        printf("%d %d %d %d\t\t%d\t\t%.4f\n", (int)x[p][0], (int)x[p][1],
               (int)x[p][2], (int)x[p][3], (int)d[p][0], o[0]);
    }
}

// XNORの実験を実行する関数
void RunExperimentXNOR(int hiddenNeurons) {
    int i, j, k, p, q = 0;
    double Error = DBL_MAX;
    double delta_o[K];
    double delta_y[J_MAX];

    J = hiddenNeurons;
    printf("\n\n---------------------------------------------\n");
    printf("Running XNOR experiment with %d hidden neurons\n", J);
    printf("---------------------------------------------\n");

    GenerateTrainingDataXNOR();
    Initialization();

    while (Error > desired_error) {
        q++;
        Error = 0;
        for (p = 0; p < n_sample; p++) {
            FindHidden(p);
            FindOutput();

            for (k = 0; k < K; k++) {
                Error += 0.5 * pow(d[p][k] - o[k], 2.0);
                delta_o[k] = (d[p][k] - o[k]) * (1 - o[k]) * o[k];
            }

            for (j = 0; j < J; j++) {
                delta_y[j] = 0;
                for (k = 0; k < K; k++) delta_y[j] += delta_o[k] * w[k][j];
                delta_y[j] = (1 - y[j]) * y[j] * delta_y[j];
            }

            for (k = 0; k < K; k++)
                for (j = 0; j < J; j++) w[k][j] += eta * delta_o[k] * y[j];

            for (j = 0; j < J; j++)
                for (i = 0; i < I; i++) v[j][i] += eta * delta_y[j] * x[p][i];
        }

        if (q % 100 == 0 || Error <= desired_error) {
            printf("Error in the %d-th learning cycle = %f\n", q, Error);
        }

        // Avoid infinite loops
        if (q > 10000) {
            printf("Max iterations reached. Stopping training.\n");
            break;
        }
    }

    PrintResult();

    // Test the network with all samples
    printf("\nTest results for XNOR with %d hidden neurons:\n", J);
    printf("Input 1\tInput 2\tDesired\tOutput\n");
    for (p = 0; p < n_sample; p++) {
        FindHidden(p);
        FindOutput();
        // XNORの場合は最初の2つの入力だけを表示
        printf("%d\t%d\t%d\t%.4f\n", (int)x[p][0], (int)x[p][1],
               (int)d[p][0], o[0]);
    }
}

/*************************************************************/
/* Generate training data for 4-bit parity check problem     */
/*************************************************************/
void GenerateTrainingDataParity(void) {
    int i, p, count;

    // Generate all possible 4-bit combinations
    for (p = 0; p < n_sample; p++) {
        // Convert p to binary and use as inputs
        x[p][0] = (p & 8) ? 1 : 0;  // 1000
        x[p][1] = (p & 4) ? 1 : 0;  // 0100
        x[p][2] = (p & 2) ? 1 : 0;  // 0010
        x[p][3] = (p & 1) ? 1 : 0;  // 0001
        x[p][4] = -1;               // Dummy input (bias)

        // Count number of ones
        count = 0;
        for (i = 0; i < 4; i++) {
            if (x[p][i] == 1) count++;
        }

        // Output is 1 if number of ones is even, 0 otherwise
        d[p][0] = (count % 2 == 0) ? 1.0 : 0.0;
    }
}

/*************************************************************/
/* Generate training data for XNOR problem                   */
/*************************************************************/
void GenerateTrainingDataXNOR(void) {
    int p;

    // For XNOR, we still use the same array structure but only the first two inputs matter
    for (p = 0; p < n_sample; p++) {
        // Set all inputs to create unique patterns
        x[p][0] = (p & 8) ? 1 : 0;
        x[p][1] = (p & 4) ? 1 : 0;
        x[p][2] = (p & 2) ? 1 : 0;
        x[p][3] = (p & 1) ? 1 : 0;
        x[p][4] = -1;  // Dummy input (bias)

        // For XNOR, output is 1 if both inputs are the same, 0 otherwise
        // We only use the first two inputs for the XNOR operation
        d[p][0] = ((x[p][0] == x[p][1]) ? 1.0 : 0.0);
    }
}

/*************************************************************/
/* Initialization of the connection weights                  */
/*************************************************************/
void Initialization(void) {
    int i, j, k;

    randomize();
    for (j = 0; j < J; j++)
        for (i = 0; i < I; i++) v[j][i] = frand() - 0.5;

    for (k = 0; k < K; k++)
        for (j = 0; j < J; j++) w[k][j] = frand() - 0.5;
}

/*************************************************************/
/* Find the output of the hidden neurons                     */
/*************************************************************/
void FindHidden(int p) {
    int i, j;
    double temp;

    for (j = 0; j < J - 1; j++) {
        temp = 0;
        for (i = 0; i < I; i++) temp += v[j][i] * x[p][i];
        y[j] = sigmoid(temp);
    }
    y[J - 1] = -1;  // Bias for the output layer
}

/*************************************************************/
/* Find the actual outputs of the network                    */
/*************************************************************/
void FindOutput(void) {
    int j, k;
    double temp;

    for (k = 0; k < K; k++) {
        temp = 0;
        for (j = 0; j < J; j++) temp += w[k][j] * y[j];
        o[k] = sigmoid(temp);
    }
}

/*************************************************************/
/* Print out the final result                                */
/*************************************************************/
void PrintResult(void) {
    int i, j, k;

    printf("\n");
    printf("The connection weights in the output layer:\n");
    for (k = 0; k < K; k++) {
        for (j = 0; j < J; j++) printf("%5f ", w[k][j]);
        printf("\n");
    }

    printf("\n");
    printf("The connection weights in the hidden layer:\n");
    for (j = 0; j < J - 1; j++) {
        for (i = 0; i < I; i++) printf("%5f ", v[j][i]);
        printf("\n");
    }
}