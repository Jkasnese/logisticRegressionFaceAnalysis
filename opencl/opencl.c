// Extracted from:
// https://mmlind.github.io/Using_Logistic_Regression_to_solve_MNIST/

/**
 * @file opencl.c
 * @author Manuella Vieira e Guilherme Lopes
 * @date 01/02/2019

Trabalho eh: Variar num de imagens, num de epocas de treino, num de computing units. Comparar com CPU somente o basico.

 */
#define _GNU_SOURCE

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<CL/cl.h>

#define TOTAL_SAMPLES 5481
#define TRAINING_SAMPLES 4487
#define TEST_SAMPLES 994
#define NUM_PIXELS 16385 // Num pixels + 1 (bias)

// #define NODES 3

const float learning_rate = 0.01; /**< Constant that holds the learning rate */

int main(int argc, char *argv[]){

    double start_time, setup_time, opencl_overhead_time, end_time, persistency_time;

    start_time = MPI_Wtime();
    
    int num_of_epochs = atoi(argv[1]); 
    int num_of_samples = atoi(argv[2]);
    int num_of_cu = atoi(argv[3]);    

    const float update = learning_rate / num_of_samples;

    // Data holders for training/testing.
        // CPU variables
        float *training, *test, *labels_train, *labels_test, *weights;

        // GPU variables
        cl_mem d_training, d_test, d_labels_train, d_labels_test, d_weights;

    // Metrics variables (GPU)    
    cl_mem d_test_accuracy, d_precision, d_recall, d_fone;
    // (CPU)
    float test_accuracy, precision, recall, fone;

    // CPU variables to hold training metrics from each epoch
    float *accuracies, *losses;

    weights = (float *)malloc(NUM_PIXELS*sizeof(float));

    // Generate array to hold hypothesis results:
    // Mudar esses mallocs pra o create_buffer_cl
//    hypothesis = (float *) malloc (num_of_samples*sizeof(float));
//    gradient = (float *) malloc (NUM_PIXELS*sizeof(float));

    // Training variables for root:
    float pixel;
        // Metrics holders
    accuracies = (float *)malloc((num_of_epochs)*sizeof(float));
    losses = (float *)malloc((num_of_epochs)*sizeof(float));
    
    training = (float *)malloc((num_of_samples * NUM_PIXELS)*sizeof(float));
    test = (float *)malloc((TEST_SAMPLES * NUM_PIXELS)*sizeof(float));

    labels_train = (float *)malloc((num_of_samples)*sizeof(float));
    labels_test = (float *)malloc((TEST_SAMPLES)*sizeof(float));
    
    // IO variables. Reading file variables.
    char train_filename[] = "fold_training_out.csv", test_filename[] = "fold_test_out.csv";
    int buffer_size = 100000, gender_female = 0, gender_male = 0;
    char *charbuffer, *gender, *usage, *char_pixels, *temp_pixels;

    FILE* train_images = fopen(train_filename, "r");
    FILE* test_images = fopen(test_filename, "r");
    int i_train = 0, i_test = 0, i = 0, j = 0;
    int offset = 0; 
    int is_training; // 1 == training, 0 == test

    charbuffer = (char *)malloc(buffer_size*sizeof(char));
    temp_pixels = (char *)malloc(2*sizeof(float));

    /** - Reads the training data from .csv file and stores the labels and pixel values in arrays */
    while(fgets(charbuffer, buffer_size, train_images) != NULL) {
        if(i_train <= num_of_samples){
            gender = strtok(charbuffer, ",");

            char_pixels = strtok(NULL, ",");


                if (strcmp(gender, "0") == 0){
                        labels_train[i_train] = 0;
                        gender_female++;
                }
                else{
                        labels_train[i_train] = 1;
                        gender_male++;
                }

                temp_pixels = strtok(char_pixels, " ");

                for (j = 0; j < (NUM_PIXELS-1); j++){
                    pixel = atof(temp_pixels);
                        offset = i_train*NUM_PIXELS + j;
                        training[offset] = pixel/255.0;

                    temp_pixels = strtok(NULL, " ");
                }

                    training[i_train*NUM_PIXELS + j] = 1;
                    i_train++;
            }
            else{
                break;
            }
        }


    /** - Reads data from test .csv file and stores the labels and pixel values in arrays */
    while(fgets(charbuffer, buffer_size, test_images) != NULL) {

            gender = strtok(charbuffer, ",");
            char_pixels = strtok(NULL, ",");


                if (strcmp(gender, "0") == 0){
                        labels_test[i_test] = 0;
                        gender_female++;
                }
                else{
                        labels_test[i_test] = 1;
                        gender_male++;
                }

                temp_pixels = strtok(char_pixels, " ");

                for (j = 0; j < (NUM_PIXELS-1); j++){
                    pixel = atof(temp_pixels);

                        offset = i_test*NUM_PIXELS + j;
                        test[offset] = pixel/255.0;

                    temp_pixels = strtok(NULL, " ");
                }

                    test[i_test*NUM_PIXELS + j] = 1;
                    i_test++;
          }

    /** - Closes the files after they have been read */
    fclose(train_images);
    fclose(test_images);

    /** - Parallelizes the loop for initializing weight values */
    for (int i=0; i<NUM_PIXELS; i++){
        weights[i] =  ( (rand() % 100) / 292.0) - 0.175; //>> 2 fica quanto mais rápido?
    }

    setup_time = MPI_Wtime(); // clock();


    // Kernel function, to be computed @ GPU
    const char *KernelSource = "\n" \
    "__kernel void train(       \n" \
    "   __global float* training,      \n" \
    "   __global float* test,          \n" \
    "   __global float* labels_train,  \n" \
    "   __global float* labels_test,   \n" \
    "   __global float* weights,        \n" \
    "   __global float test_accuracy \n" \
    "   __global float precision \n" \
    "   __global float recall \n" \
    "   __global float fone \n" \
    "   const int num_test_samples,  \n" \
    "   const int num_of_epochs, \n" \
    "   const int num_of_training_imgs)      \n" \
    "{ \n" \
    "    for (int epochs=0; epochs<num_of_epochs; epochs++){  \n" \
    "    // Zeroing epoch stats \n" \
    "    right_answers = 0; \n" \ // Vai usar essas metricas de cada epoca?
    "    loss = 0; \n" \ // Vai usar essas metricas de cada epoca?
    " \n" \
    "    // Zeroing gradients from previous epoch \n" \
    "    for (int i = 0; i < NUM_PIXELS; ++i) \n" \
    "    { \n" \
    "        gradient[i] = 0; \n" \
    "    } \n" \
    " \n" \
    "    for (long r=0; r<num_of_training_imgs; r++){ \n" \
    "        r_numpixels = r*NUM_PIXELS; \n" \
    "        temp = 0; \n" \
    "        for (long x=0; x<NUM_PIXELS; x++){ \n" \
    "            temp += *(training + (r_numpixels)+x) * *(weights + x); \n" \
    "        } \n" \
    "        /** - Calculates logistic hypothesis */ \n" \
    "        temp = 1 / (1 + (exp( -1.0 * temp)) ); \n" \
    " \n" \
    "         /** - Computes accuracy on training set */ \n" \
    "        if (labels_train[r] == 0.0){ \n" \
    "            if (temp < 0.5) \n" \
    "                right_answers++; \n" \
    "        } else { \n" \
    "            if (!(temp < 0.5)) \n" \
    "                right_answers++; \n" \
    "        } \n" \
    " \n" \
    "        /** - Computes loss function */ \n" \
    "        aux = labels_train[r]*log(temp) + (1 - labels_train[r])*log(1-temp); \n" \
    "        loss += aux; // Acelera se trocar por if/else dos labels? \n" \
    " \n" \
    "        /** - Computes the difference between label and hypothesis */ \n" \
    "        aux = labels_train[r] - temp; \n" \
    " \n" \
    "        /** - Computes current gradient */ \n" \
    "        r_numpixels = r*NUM_PIXELS; \n" \
    "        for (long x=0; x<NUM_PIXELS; x++){ \n" \
    "            gradient[x] += training[r_numpixels + x] * aux; \n" \
    "        } \n" \
    "    } \n" \
    " \n" \
    "    /** - Updates weights */ \n" \
    "    for (int i=0; i<NUM_PIXELS; i++){ \n" \
    "        weights[i] += update * gradient[i]; \n" \
    "    } \n" \
    " \n" \
    "    /** - Saves epoch metrics to be plotted later */ \n" \ // Isto so eh util se transferir os dados da GPU pra CPU depois de cada epoca.
    "   } \n" \
    " \n" \
    "// CALCULATE TEST METRICS \n" \
    " // Zeroing variables to hold metrics stats: \n" \
    "right_answers = 0; \n" \
    "int fp = 0, tp = 0, tn = 0, fn = 0; \n" \
    " \n" \
    "/** - Generate hypothesis values for the test set */ \n" \
    "for (int r=0; r<TEST_SAMPLES; r++){ \n" \
    "    r_numpixels = r*NUM_PIXELS; \n" \
    "    temp = 0; \n" \
    "    for (long x=0; x<NUM_PIXELS; x++){ \n" \
    "        temp += *(test + r_numpixels+x) * weights[x]; \n" \
    "    } \n" \
    "     \n" \
    "    // Calculate logistic hypothesis \n" \
     "   temp = 1 / (1 + (exp( -1.0 * temp)) ); \n" \
    " \n" \
    "    // Compute the difference between label and hypothesis & \n" \
    "    //  accuracy on training set & \n" \
    "    //  loss function & \n" \
    "    //  metrics (accuracy, precision, recall, f1) \n" \
    "    if (labels_test[r] == 1.0){ \n" \
    "        if (temp < 0.5){ \n" \
    "            // FP \n" \
    "            fp++; \n" \
    "        } else { \n" \
    "            // TP \n" \
    "            tp++; \n" \
    "        } \n" \
    "    } else { \n" \
    "        if (temp < 0.5){ \n" \
    "            // TN \n" \
    "            tn++; \n" \
    "        } else { \n" \
    "             // FN \n" \
    "            fn++; \n" \
    "         } \n" \
    "    } \n" \
    " \n" \
    "    test_accuracy = ((float) (tp + tn))/ num_test_samples; \n" \
    "    precision = ((float) tp) / (tp+fp); \n" \
    "    recall = ((float) tp) / (tp + fn); \n" \
    "    fone = 2*((precision*recall) / (precision + recall)); \n" \
    "} \n" \

    
    end_time = MPI_Wtime(); // clock();

    /** - Writes metrics (accuracy, loss, precision, recall and F1 score) to files */ 
    FILE* ftest = fopen("test_metrics.txt", "w");
    FILE* execution_times = fopen("execution_times", "w");

    fprintf(ftest, "%s %f\n%s %f\n%s %f\n%s %f\n", "accuracy ", test_accuracy, "precision ", precision, "recall ", recall, "f1 ", fone);
    
    fprintf(execution_times, "%s\t%0.9f\n", "Setup Serial time: ", (setup_time - start_time) );
    fprintf(execution_times, "%s\t%0.9f\n", "OpenCL Overhead time: ", ( (opencl_overhead_time - setup_time)) + (persistency_time - end_time) );
    fprintf(execution_times, "%s\t%0.9f\n", "Training time: ", (end_time - opencl_overhead_time) );
    fprintf(execution_times, "%s\t%0.9f\n", "Total time: ", (persistency_time - start_time) );

    fclose(ftest);
    fclose(execution_times);
}
