#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <cblas.h> //You need to install cblas libary

struct Data
{
    int num_of_input;
    int num_of_output;
    int num_of_data;
    float *X;
    float *Y;
};

struct layer
{
    int neu_n;
    float *A;
    float *W;
    float *B;

    float *dA;
    float *dW;
    float *dB;
};

struct neu_net
{
    struct layer *layers;
    int num_of_lay;
    int *neu_in_lay;
    float learn_rate;
    int all_neu;
    int all_weight;
    char type;
};

void Set_Net(struct neu_net *, int, int *, float);
void Resize_Net(struct neu_net *, int);
void Clean_Net(struct neu_net *);
void Set_Data(struct Data *, int, int, int);
void Get_Data(struct Data *);
void Get_XY(float * x, float * y);  // Do it by youself, load Input to x and Output to y
void Choose_Data(struct Data *, struct Data *);
void Clean_Data(struct Data *);
void Run_Net_X(struct neu_net *, struct Data *);
float Cost_of_X(struct neu_net *, struct Data *);
void Refine_Net(struct neu_net *, struct Data *);
float *Run_Net(struct neu_net *, float *);
void Show_Actv_i(struct neu_net *, int);
void Show_Weight_i(struct neu_net *, int);
void Show_Bias_i(struct neu_net *, int);
void Dump_Net(struct neu_net *);
void Load_Net(struct neu_net *);

void main()
{
    const int num_of_lay = 5;
    int neu_in_lay[] = {784, 28, 28, 28, 10};
    const float learn_rate = 0.15;
    const float num_of_data = 10000;
    const int num_of_x = 100;
    const int Generation = 500;

    struct neu_net Net;
    Set_Net(&Net, num_of_lay, neu_in_lay, learn_rate);
    //Load_Net(&Net);
    const int num_of_input = Net.neu_in_lay[0];
    const int num_of_output = Net.neu_in_lay[Net.num_of_lay - 1];

    struct Data Data_all, Datax;
    Set_Data(&Data_all, num_of_input, num_of_output, num_of_data);
    Set_Data(&Datax, num_of_input, num_of_output, num_of_x);
    Get_Data(&Data_all);

    int i;
    for (i = 0; i < Generation; i++)
    {
        Choose_Data(&Data_all, &Datax);
        Refine_Net(&Net, &Datax);
        printf("Geeration:%05d     CostAll:%0.3f\n", i, Cost_of_X(&Net, &Datax));
    }
    printf("\n");
    Show_Actv_i(&Net, 0);
    Show_Bias_i(&Net, 0);
    Show_Weight_i(&Net, 0);

    Dump_Net(&Net);

    Clean_Net(&Net);
    Clean_Data(&Data_all);
    Clean_Data(&Datax);
    return;
}

void Set_Net(struct neu_net *Net, int num_of_lay, int *neu_in_lay, float learn_rate)
{
    const char final_lay_fuction = 's';

    Net->num_of_lay = num_of_lay;
    Net->neu_in_lay = neu_in_lay;
    Net->learn_rate = learn_rate;
    int i, j, all_neu = 0, all_weight = 0;
    for (i = 0; i < num_of_lay; i++)
    {
        all_neu += neu_in_lay[i];
        all_weight += (i) ? neu_in_lay[i - 1] * neu_in_lay[i] : 0;
    }
    Net->all_neu = all_neu;
    Net->all_weight = all_weight;
    Net->type = final_lay_fuction;

    Net->layers = (struct layer *)malloc(num_of_lay * sizeof(struct layer));
    struct layer *layp = Net->layers;
    layp->neu_n = neu_in_lay[0];
    layp->A = (float *)calloc(all_neu, sizeof(float));
    layp->W = (float *)calloc(all_weight, sizeof(float));
    layp->B = (float *)calloc(all_neu, sizeof(float));
    layp->dA = (float *)calloc(all_neu, sizeof(float));
    layp->dW = (float *)calloc(all_weight, sizeof(float));
    layp->dB = (float *)calloc(all_neu, sizeof(float));
    if (layp->A == NULL || layp->W == NULL || layp->B == NULL || layp->dA == NULL || layp->dW == NULL || layp->dB == NULL)
        abort();

    int weight_in_previous, weight_in_this = 0, neu_in_Previous, neu_in_this = neu_in_lay[0];
    for (i = 1; i < num_of_lay; i++)
    {
        layp++;
        neu_in_Previous = neu_in_this;
        neu_in_this = neu_in_lay[i];
        weight_in_previous = weight_in_this;
        weight_in_this = neu_in_Previous * neu_in_this;

        layp->neu_n = neu_in_this;
        layp->A = (layp - 1)->A + neu_in_Previous;
        layp->W = (layp - 1)->W + weight_in_previous;
        layp->B = (layp - 1)->B + neu_in_Previous;
        layp->dA = (layp - 1)->dA + neu_in_Previous;
        layp->dW = (layp - 1)->dW + weight_in_previous;
        layp->dB = (layp - 1)->dB + neu_in_Previous;
    }
    srand(1);
    for (j = 0; j < all_weight; j++)
        Net->layers->W[j] = rand() / (RAND_MAX + 1.0) - 0.5;
}

void Resize_Net(struct neu_net *Net, int num_of_x)
{
    static struct neu_net *prenet = NULL;
    static int prex = 0;

    if (Net != prenet || num_of_x != prex)
    {
        prenet = Net;
        prex = num_of_x;

        // from Net
        const int num_of_lay = Net->num_of_lay;
        const int *neu_in_lay = Net->neu_in_lay;

        // resize memory
        Net->layers->A = (float *)realloc(Net->layers->A, Net->all_neu * num_of_x * sizeof(float));
        Net->layers->dA = (float *)realloc(Net->layers->dA, Net->all_neu * num_of_x * sizeof(float));
        if (Net->layers->A == NULL || Net->layers->dA == NULL)
            abort();

        // redirect pointers
        int i;
        struct layer *layp = Net->layers;
        int neu_in_previous;
        for (i = 1; i < Net->num_of_lay; i++)
        {
            layp++;
            neu_in_previous = neu_in_lay[i - 1];

            layp->A = (layp - 1)->A + neu_in_previous * num_of_x;
            layp->dA = (layp - 1)->dA + neu_in_previous * num_of_x;
        }
    }
}

void Clean_Net(struct neu_net *Net)
{
    free(Net->layers->A);
    free(Net->layers->W);
    free(Net->layers->B);
    free(Net->layers->dA);
    free(Net->layers->dW);
    free(Net->layers->dB);

    for (int i = 0; i < Net->num_of_lay; i++)
    {
        Net->layers[i].A = NULL;
        Net->layers[i].W = NULL;
        Net->layers[i].B = NULL;
        Net->layers[i].dA = NULL;
        Net->layers[i].dW = NULL;
        Net->layers[i].dB = NULL;
        Net->layers[i].neu_n = 0;
    }
    Net->layers = NULL;

    memset(Net->neu_in_lay, 0, Net->num_of_lay * sizeof(float));
    Net->num_of_lay = 0;
    Net->neu_in_lay = NULL;
    Net->all_neu = 0;
    Net->all_weight = 0;
    Net->learn_rate = 0;
    Net->type = '\0';
}

void Set_Data(struct Data *datap, int num_of_input, int num_of_output, int num_of_data)
{
    datap->num_of_input = num_of_input;
    datap->num_of_output = num_of_output;
    datap->num_of_data = num_of_data;
    datap->X = (float *)malloc(num_of_data * (num_of_input + num_of_output) * sizeof(float));
    datap->Y = datap->X + num_of_data * num_of_input;
}

void Get_Data(struct Data *data)
{
    const int num_of_data = data->num_of_data;
    const int num_of_input = data->num_of_input;
    const int num_of_output = data->num_of_output;
    float *X = data->X;
    float *Y = data->Y;

    int i, j;
    for (i = 0; i < num_of_data; i++)
    {
        Get_XY(X, Y);
        X += num_of_input;
        Y += num_of_output;
    }
}

void Get_XY(float *x, float *y)
{
    // Load input to x

    // Load output distribution to y

}

void Choose_Data(struct Data *data, struct Data *datax)
{
    const int num_of_x = datax->num_of_data;
    const int num_of_data = data->num_of_data;
    const int num_of_input = data->num_of_input;
    const int num_of_output = data->num_of_output;
    if (num_of_input != datax->num_of_input || num_of_output != datax->num_of_output)
        abort();

    int i, j;
    for (i = 0; i < num_of_x; i++)
    {
        j = rand() % num_of_data;
        cblas_scopy(num_of_input, data->X + j * num_of_input, 1, datax->X + i * num_of_input, 1);
        cblas_scopy(num_of_output, data->Y + j * num_of_output, 1, datax->Y + i * num_of_output, 1);
    }
}

void Clean_Data(struct Data *data)
{
    data->num_of_input = 0;
    data->num_of_output = 0;
    data->num_of_data = 0;
    free(data->X);
    data->X = NULL;
    data->Y = NULL;
}

void Run_Net_X(struct neu_net *Net, struct Data *data)
{
    // from Net
    const int num_of_lay = Net->num_of_lay;
    const int num_of_input = Net->neu_in_lay[0];
    const int num_of_output = Net->neu_in_lay[Net->num_of_lay - 1];

    // from data
    const int num_of_x = data->num_of_data;
    const float *X = data->X;

    // initial
    Resize_Net(Net, num_of_x);
    cblas_scopy(num_of_input * num_of_x, X, 1, Net->layers->A, 1);

    // recursive
    struct layer *layp = Net->layers;
    int i, j;
    int neu_in_this = num_of_input, neu_in_previous;
    for (i = 1; i < num_of_lay - 1; i++)
    {
        layp++;
        neu_in_previous = neu_in_this;
        neu_in_this = layp->neu_n;

        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, neu_in_this, num_of_x, neu_in_previous, 1, layp->W, neu_in_previous, (layp - 1)->A, neu_in_previous, 0, layp->A, neu_in_this);

        for (j = neu_in_this * num_of_x; j--;)
            if ((layp->A[j] += layp->B[j % neu_in_this]) < 0)
                layp->A[j] = 0;
    }

    // final layer
    layp++;
    neu_in_previous = neu_in_this;
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, num_of_output, num_of_x, neu_in_previous, 1, layp->W, neu_in_previous, (layp - 1)->A, neu_in_previous, 0, layp->A, num_of_output);
    if (Net->type == 's')
    {
        for (i = num_of_output * num_of_x; i--;)
            layp->A[i] = 1. / (1. + expf(-(layp->A[i] + layp->B[i % num_of_output])));
    }
    else if (Net->type == 'r')
    {
        for (i = num_of_output * num_of_x; i--;)
            layp->A[i] += layp->B[i % num_of_output];
    }
    else
        abort();
}

float Cost_of_X(struct neu_net *Net, struct Data *data)
{
    // from Net
    const float *outputA = Net->layers[Net->num_of_lay - 1].A;

    // from data
    const int num_of_x = data->num_of_data;
    const int num_of_output = data->num_of_output;
    const float *Y = data->Y;

    float *C = (float *)malloc(num_of_output * num_of_x * sizeof(float));
    cblas_scopy(num_of_output * num_of_x, Y, 1, C, 1);
    cblas_saxpy(num_of_output * num_of_x, -1, outputA, 1, C, 1);
    float cost = cblas_snrm2(num_of_output * num_of_x, C, 1);
    cost /= num_of_output * num_of_x * 2;
    free(C);
    return cost;
}

void Refine_Net(struct neu_net *Net, struct Data *data)
{
    // from Net
    const int num_of_lay = Net->num_of_lay;
    const int *neu_in_lay = Net->neu_in_lay;
    const int num_of_output = Net->neu_in_lay[Net->num_of_lay - 1];
    const float learn_rate = Net->learn_rate;
    const int all_neu = Net->all_neu;
    const int all_weight = Net->all_weight;

    // from data
    const int num_of_x = data->num_of_data;
    const float *Y = data->Y;

    // initial
    int i, j;
    struct layer *layp = Net->layers + num_of_lay - 1;
    int neu_in_previous = (layp - 1)->neu_n;

    // A
    Run_Net_X(Net, data);

    // dA
    cblas_scopy(num_of_output * num_of_x, Y, 1, layp->dA, 1);
    cblas_saxpy(num_of_output * num_of_x, -1, layp->A, 1, layp->dA, 1);

    // R
    float *R0 = (float *)calloc(all_neu * num_of_x, sizeof(float));
    float *Ri = R0 + (all_neu - num_of_output) * num_of_x;
    const float rdividex = learn_rate / num_of_x;
    if (Net->type == 's')
    {
        for (i = num_of_output * num_of_x; i--;)
            Ri[i] = rdividex / num_of_output * layp->dA[i] * layp->A[i] * (1 - layp->A[i]);
    }
    else if (Net->type == 'r')
    {
        cblas_saxpy(num_of_output * num_of_x, rdividex / num_of_output, layp->dA, 1, Ri, 1);
    }
    else
        abort();

    // dB
    cblas_scopy(num_of_output, Ri, 1, layp->dB, 1);
    for (i = num_of_x; --i;)
        cblas_saxpy(num_of_output, 1, Ri + i * num_of_output, 1, layp->dB, 1);

    // dW
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, num_of_output, neu_in_previous, num_of_x, 1, Ri, num_of_output, (layp - 1)->A, neu_in_previous, 0, layp->dW, neu_in_previous);

    int neu_in_next, neu_in_this = num_of_output;
    for (i = num_of_lay - 1; --i;)
    {
        layp--;
        neu_in_next = neu_in_this;
        neu_in_this = neu_in_previous;
        neu_in_previous = (layp - 1)->neu_n;

        // dA
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, neu_in_this, num_of_x, neu_in_next, num_of_x, (layp + 1)->W, neu_in_this, Ri, neu_in_next, 0, layp->dA, neu_in_this);

        // R
        Ri -= neu_in_this * num_of_x;
        cblas_saxpy(neu_in_this * num_of_x, rdividex / neu_in_this, layp->dA, 1, Ri, 1);
        for (j = neu_in_this * num_of_x; j--;)
            if (!layp->A[j])
                Ri[j] = 0;

        // dB
        cblas_scopy(neu_in_this, Ri, 1, layp->dB, 1);
        for (j = num_of_x; --j;)
            cblas_saxpy(neu_in_this, 1, Ri + j * neu_in_this, 1, layp->dB, 1);

        // dW
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, neu_in_this, neu_in_previous, num_of_x, 1, Ri, neu_in_this, (layp - 1)->A, neu_in_previous, 0, layp->dW, neu_in_previous);
    }

    // W B
    cblas_saxpy(all_weight, 1, Net->layers->dW, 1, Net->layers->W, 1);
    cblas_saxpy(all_neu, 1, Net->layers->dB, 1, Net->layers->B, 1);

    free(R0);
}

float *Run_Net(struct neu_net *Net, float *X)
{
    const int num_of_lay = Net->num_of_lay;
    const int *neu_in_lay = Net->neu_in_lay;
    const int all_neu = Net->all_neu;
    const int num_of_input = Net->neu_in_lay[0];
    const int num_of_output = Net->neu_in_lay[Net->num_of_lay - 1];

    struct layer *layp = Net->layers;
    int neu_in_this = num_of_input;

    cblas_scopy(num_of_input, X, 1, layp->A, 1);

    layp++;
    int i, j, neu_in_previous;
    for (i = 1; i < num_of_lay - 1; i++, layp++)
    {
        neu_in_previous = neu_in_this;
        neu_in_this = layp->neu_n;
        layp->A = (layp - 1)->A + neu_in_previous;

        cblas_scopy(neu_in_this, layp->B, 1, layp->A, 1);
        cblas_sgemv(CblasRowMajor, CblasNoTrans, neu_in_this, neu_in_previous, 1, layp->W, neu_in_previous, (layp - 1)->A, 1, 1, layp->A, 1);

        for (j = neu_in_this; j--;)
            if (layp->A[j] < 0)
                layp->A[j] = 0;
    }

    neu_in_previous = neu_in_this;
    layp->A = (layp - 1)->A + neu_in_previous;

    cblas_scopy(neu_in_this, layp->B, 1, layp->A, 1);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, neu_in_this, neu_in_previous, 1, layp->W, neu_in_previous, (layp - 1)->A, 1, 1, layp->A, 1);

    if (Net->type == 's')
        for (i = num_of_output; i--;)
            layp->A[i] = 1. / (1. + expf(-(layp->A[i])));
    else
        for (i = neu_in_this; i--;)
            if (layp->A[i] < 0)
                layp->A[i] = 0;

    return Net->layers->A;
}

void Show_Actv_i(struct neu_net *Net, int xi)
{
    const int num_of_lay = Net->num_of_lay;
    const int *neu_in_lay = Net->neu_in_lay;
    const int num_of_input = Net->neu_in_lay[0];
    const int num_of_output = Net->neu_in_lay[Net->num_of_lay - 1];
    const int all_neu = Net->all_neu;

    printf("------------------------------------------\n\n");
    printf("Actv:\n");
    int i, j;
    for (i = 0; i < num_of_lay; i++)
    {
        for (j = 0; j < neu_in_lay[i]; j++)
        {
            printf("%+05.3f ", Net->layers[i].A[j + xi * neu_in_lay[i]]);
        }
        printf("\n");
    }
    printf("\n");
}

void Show_Bias_i(struct neu_net *Net, int xi)
{
    const int num_of_lay = Net->num_of_lay;
    const int *neu_in_lay = Net->neu_in_lay;
    const int num_of_input = Net->neu_in_lay[0];
    const int num_of_output = Net->neu_in_lay[Net->num_of_lay - 1];
    const int all_neu = Net->all_neu;

    printf("------------------------------------------\n\n");
    printf("Bias:\n");
    int i, j;
    for (i = 0; i < num_of_lay; i++)
    {
        for (j = 0; j < neu_in_lay[i]; j++)
        {
            printf("%+05.3f ", Net->layers[i].B[j + xi * neu_in_lay[i]]);
        }
        printf("\n");
    }
    printf("\n");
}

void Show_Weight_i(struct neu_net *Net, int xi)
{
    const int num_of_lay = Net->num_of_lay;
    const int *neu_in_lay = Net->neu_in_lay;
    const int num_of_input = Net->neu_in_lay[0];
    const int num_of_output = Net->neu_in_lay[Net->num_of_lay - 1];
    const int all_neu = Net->all_neu;

    printf("------------------------------------------\n\n");
    printf("Weight:\n");
    int i, j, k;
    for (i = 1; i < num_of_lay; i++)
    {
        for (j = 0; j < neu_in_lay[i]; j++)
        {
            for (k = 0; k < neu_in_lay[i - 1]; k++)
                printf("%+05.3f ", Net->layers[i].W[k + j * neu_in_lay[i - 1]]);
            printf("\n");
        }
        printf("\n");
    }
}

void Dump_Net(struct neu_net *Net)
{
    const int num_of_lay = Net->num_of_lay;
    const int *neu_in_lay = Net->neu_in_lay;
    const float learn_rate = Net->learn_rate;
    const int all_neu = Net->all_neu;
    const int all_weight = Net->all_weight;
    const char type = Net->type;

    FILE *fp = fopen("neu_net.txt", "w");
    fprintf(fp, "num_of_lay = %d\n", num_of_lay);
    fprintf(fp, "neu_in_lay = ");
    for (int i = 0; i < num_of_lay; i++)
        fprintf(fp, "%d ", neu_in_lay[i]);
    fprintf(fp, "\n");
    fprintf(fp, "learn_rate = %f\n", learn_rate);
    fprintf(fp, "all_neu = %d\n", all_neu);
    fprintf(fp, "all_weight = %d\n", all_weight);
    fprintf(fp, "type = %c\n", type);
    fprintf(fp, "weight = \n");
    for (int i = 0; i < all_weight; i++)
        fprintf(fp, "%f ", Net->layers->W[i]);
    fprintf(fp, "\n");
    fprintf(fp, "bias = \n");
    for (int i = 0; i < all_neu; i++)
        fprintf(fp, "%f ", Net->layers->B[i]);
    fclose(fp);
}

void Load_Net(struct neu_net *Net)
{
    int i;
    int success = 1,a;
    FILE *fp = fopen("neu_net.txt", "r");
    success *= fscanf(fp, "num_of_lay = %d\n", &(Net->num_of_lay));
    a = fscanf(fp, "neu_in_lay = ");
    for (i = 0; i < Net->num_of_lay; i++)
        success *= fscanf(fp, "%d ", Net->neu_in_lay + i);
    a = fscanf(fp, "\n");
    success *= fscanf(fp, "learn_rate = %f\n", &(Net->learn_rate));
    success *= fscanf(fp, "all_neu = %d\n", &(Net->all_neu));
    success *= fscanf(fp, "all_weight = %d\n", &(Net->all_weight));
    success *= fscanf(fp, "type = %c\n", &(Net->type));
    a = fscanf(fp, "weight = \n");
    for (i = 0; i < Net->all_weight; i++)
        success *= fscanf(fp, "%f ", Net->layers->W + i);
    a = fscanf(fp, "\n");
    a = fscanf(fp, "bias = \n");
    for (i = 0; i < Net->all_neu; i++)
        success *= fscanf(fp, "%f ", Net->layers->B + i);
    fclose(fp);
    if (!success)
        abort();
}
