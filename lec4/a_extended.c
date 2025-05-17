#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define I 4 /* 入力次元数 */
#define M 3 /* ニューロン数 */
#define P 150 /* パターン数 */
#define alpha 0.5 /* 学習率 */
#define n_update 20

double w[M][I];
double x[P][I];

/* Iris CSV 読み込み */
void load_iris(const char *filename)
{
    FILE *fp = fopen(filename, "r");
    if (!fp)
    {
        perror("File open error"); /* エラー報告 */
        exit(EXIT_FAILURE);
    }
    char line[256];
    /* ヘッダー行スキップ */
    fgets(line, sizeof(line), fp); /* 1行目を読み捨て */
    int p = 0;
    while (p < P && fgets(line, sizeof(line), fp))
    {
        if (line[0] == '\n' || line[0] == '\r')
            continue;
        char *tok = strtok(line, ",");
        for (int i = 0; i < I; i++)
        {
            if (!tok)
            {
                fprintf(stderr, "Malformed line\n");
                exit(EXIT_FAILURE);
            }
            x[p][i] = atof(tok);
            tok = strtok(NULL, ",");
        }
        p++;
    }
    fclose(fp);
}

void PrintResult(int q)
{
    int m, i;

    printf("\n\n");
    printf("Results in the %d-th iteration: \n", q);
    for (m = 0; m < M; m++)
    {
        for (i = 0; i < I; i++)
            printf("%5f ", w[m][i]);
        printf("\n");
    }
    printf("\n\n");
}

/*************************************************************/
/* The main program                                          */
/*************************************************************/
int main()
{
    int m, m0, i, p, q;
    double norm, s, s0;
    load_iris("iris.csv"); /* Iris CSV 読み込み */

    /* Initialization of the connection weights */

    for (m = 0; m < M; m++)
    {
        norm = 0;
        for (i = 0; i < I; i++)
        {
            w[m][i] = (double)(rand() % 10001) / 10000.0 - 0.5;
            norm += w[m][i] * w[m][i];
        }
        norm = sqrt(norm);
        for (i = 0; i < I; i++)
            w[m][i] /= norm;
    }
    PrintResult(0);

    /* Unsupervised learning */

    for (q = 0; q < n_update; q++)
    {
        for (p = 0; p < P; p++)
        {
            s0 = 0;
            for (m = 0; m < M; m++)
            {
                s = 0;
                for (i = 0; i < I; i++)
                    s += w[m][i] * x[p][i];
                if (s > s0)
                {
                    s0 = s;
                    m0 = m;
                }
            }

            for (i = 0; i < I; i++)
                w[m0][i] += alpha * (x[p][i] - w[m0][i]);

            norm = 0;
            for (i = 0; i < I; i++)
                norm += w[m0][i] * w[m0][i];
            norm = sqrt(norm);
            for (i = 0; i < I; i++)
                w[m0][i] /= norm;
        }
        PrintResult(q);
    }

    /* Classify the training patterns */

    for (p = 0; p < P; p++)
    {
        s0 = 0;
        for (m = 0; m < M; m++)
        {
            s = 0;
            for (i = 0; i < I; i++)
                s += w[m][i] * x[p][i];
            if (s > s0)
            {
                s0 = s;
                m0 = m;
            }
        }
        printf("Pattern[%d] belongs to %d-th class\n", p+1, m0+1);
    }
}
