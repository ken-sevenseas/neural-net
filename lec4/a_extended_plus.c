#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define I 4 /* 入力次元数 */
#define M 3 /* ニューロン数 */
#define P 150 /* パターン数 */
#define alpha 0.4 /* 学習率 */
#define n_update 20

double w[M][I];
double x[P][I];
int true_labels[P]; // 正解ラベルを保存する配列

/* データを正規化する関数 */
void normalize_data() {
    double mean[I] = {0};
    double std_dev[I] = {0};
    int i, p;

    // 平均を計算
    for (i = 0; i < I; i++) {
        for (p = 0; p < P; p++) {
            mean[i] += x[p][i];
        }
        mean[i] /= P;
    }

    // 標準偏差を計算
    for (i = 0; i < I; i++) {
        for (p = 0; p < P; p++) {
            std_dev[i] += (x[p][i] - mean[i]) * (x[p][i] - mean[i]);
        }
        std_dev[i] = sqrt(std_dev[i] / P);
    }

    // データを正規化
    for (i = 0; i < I; i++) {
        if (std_dev[i] > 0) {  // ゼロ除算を防止
            for (p = 0; p < P; p++) {
                x[p][i] = (x[p][i] - mean[i]) / std_dev[i];
            }
        }
    }
}

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
        // 5列目に種類情報があると仮定して読み取る
        if (tok) {
            // Irisデータセットの場合、最後の列はラベル
            if (strstr(tok, "setosa")) true_labels[p] = 0;
            else if (strstr(tok, "versicolor")) true_labels[p] = 1;
            else if (strstr(tok, "virginica")) true_labels[p] = 2;
            else true_labels[p] = -1;  // 不明なラベル
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

/* 分類結果を評価する関数 */
void evaluate_clustering() {
    int cluster_counts[M][3] = {0}; // クラスタごとの各種類のカウント
    int p, m, m0;
    double s, s0;

    // 各パターンをクラスタに分類し、カウント
    for (p = 0; p < P; p++) {
        // 最近傍クラスタを最小二乗距離で決定
        double best_dist = 1e9;
        for (m = 0; m < M; m++) {
            double dist = 0;
            for (int i = 0; i < I; i++)
                dist += (x[p][i] - w[m][i]) * (x[p][i] - w[m][i]);
            if (dist < best_dist) {
                best_dist = dist;
                m0 = m;
            }
        }
        // クラスタm0に分類された種類true_labels[p]をカウント
        if (true_labels[p] >= 0 && true_labels[p] < 3) {
            cluster_counts[m0][true_labels[p]]++;
        }
    }

    // 結果の表示
    printf("\n分類結果の集計:\n");
    for (m = 0; m < M; m++) {
        printf("クラスタ %d: setosa = %d, versicolor = %d, virginica = %d\n", 
               m+1, cluster_counts[m][0], cluster_counts[m][1], cluster_counts[m][2]);
    }
}

/*************************************************************/
/* The main program                                          */
/*************************************************************/
int main()
{
    int m, m0, i, p, q;
    int idx[P];  /* パターンシャッフル用インデックス配列 */
    double norm, s, s0;
    
    // 乱数シードを設定
    srand(time(NULL));
    
    load_iris("iris.csv"); /* Iris CSV 読み込み */
    normalize_data(); /* データの正規化 */

    /* Initialize index array for shuffling */
    for (p = 0; p < P; p++)
        idx[p] = p;
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
        /* Shuffle patterns before each epoch */
        /* 減衰学習率の計算 */
        double lr = alpha * (1.0 - (double)q / (n_update - 1));
        for (int k = P - 1; k > 0; k--) {
            int j = rand() % (k + 1);
            int tmp = idx[k];
            idx[k] = idx[j];
            idx[j] = tmp;
        }
        for (int pi = 0; pi < P; pi++) {
            p = idx[pi];

            // 勝者ニューロンを最小二乗距離で決定
            double best_dist = 1e9;
            for (m = 0; m < M; m++) {
                double dist = 0;
                for (i = 0; i < I; i++)
                    dist += (x[p][i] - w[m][i]) * (x[p][i] - w[m][i]);
                if (dist < best_dist) {
                    best_dist = dist;
                    m0 = m;
                }
            }

            for (i = 0; i < I; i++)
                w[m0][i] += lr * (x[p][i] - w[m0][i]);

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
        /* 勝者ニューロンを最小二乗距離で決定 */
        double best_dist = 1e9;
        for (m = 0; m < M; m++) {
            double dist = 0;
            for (i = 0; i < I; i++)
                dist += (x[p][i] - w[m][i]) * (x[p][i] - w[m][i]);
            if (dist < best_dist) {
                best_dist = dist;
                m0 = m;
            }
        }
        printf("Pattern[%d] belongs to %d-th class\n", p+1, m0+1);
    }
    
    // 分類結果の評価
    evaluate_clustering();
    
    return 0;
}
