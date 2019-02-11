// Extracted from:
// https://mmlind.github.io/Using_Logistic_Regression_to_solve_MNIST/

/**
 * @file opencl.c
 * @author Manuella Vieira e Guilherme Lopes
 * @date 01/02/2019

Trabalho eh: Variar num de imagens, num de epocas de treino, num de computing units. Comparar com CPU somente o basico.

Variar numero de work dimensions em clEnqueueNDRangeKernel. min 1 max 3

O gradiente pode ser calculado com todos os worksgroups calculando seu gradiente e depositando em algum lugar da memoria global. Blz.
E depois? Como sincronizar pra que, ao final do calculo do gradiente, seja calculado os novos pesos e entao a nova epoca de treino?

Acho que isso pode ser resolvido com child queue`s. Faz um kernel que chame outros kernels. Um loop contendo 2 kernels:
 - O primeiro kernel calcula o gradiente atraves do treino
 - O segundo kernel atualiza o gradiente
Depois de finalizados estes dois kernels, o kernel pai chama depois o kernel do teste, pra calcular as estatisticas de teste
Ou entao fazer esses enqueues no proprio host, dentro de um for que contenha o numero de epocas.

Usar atomic add nos gradientes, ja que as threads vao atualizar o gradiente em paralelo.

Fazer versao que explore shared memory e outra que nao explore (cache) pra comparar os tempos.

Usar OpenCL profiler.

Comparar operation e memory throughput com o maximo teorico do dispositivo.

Latencia da memoria global eh muito maior que da memoria local, ou shared. Talvez compense separar conjuntos de imagens
diferentes pra cada um dos CUs.

"Applications can also parameterize NDRanges based on register file size and shared memory size, which depends on the compute 
capability of the device, as well as on the number of multiprocessors and memory bandwidth of the device,
all of which can be queried using the OpenCL API" Usar CUDA Occupancy Calculator

"The number of threads per block should be chosen as a multiple of the warp size
 to avoid wasting computing resources with under-populated warps as much as possible." -> Tem um comando do opencl pra ver o warp_size.

 gcc -v -Wall -std=c99 -I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include" -I "D:\Documents\UEFS\7 - Sétimo Semetre\OpenCL_codes\C_common" -L "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\Win32" host_01.c -o host_01 "D:\Documents\GitHub\logisticRegressionFaceAnalysis\opencl\regressor_01.cl" "D:\Documents\UEFS\7 - Sétimo Semetre\OpenCL_codes\C_common\wtime.c" -lOpenCL

 */
#define _GNU_SOURCE

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif

#include "err_code.h"


#define TOTAL_SAMPLES 5481
#define TRAINING_SAMPLES 4487
#define TEST_SAMPLES 994
#define NUM_PIXELS 16385 // Num pixels + 1 (bias)

#define MAXDEVICES 6
#define EXIT_FAILURE -1

#define BLOCK_SIZE 32 // Variar


// #define NODES 3

const float learning_rate = 0.01; /**< Constant that holds the learning rate */

// Kernel function, to be computed @ GPU
extern int output_device_info(cl_device_id);


int main(int argc, char *argv[]){




    double start_time, setup_time, devices_time, opencl_scan, opencl_overhead_time, end_time, persistency_time;
    double rtime;                   // current time
    start_time = wtime();
    
    // Training hyperparameters
    int num_of_epochs = atoi(argv[1]); 
    int num_of_samples = atoi(argv[2]);
    int num_of_cu = atoi(argv[3]); 
    int num_test_samples = TEST_SAMPLES;   


    // CL devices variables
    int err;
    int global;
    cl_device_id     device_id = NULL;      // device id
    cl_platform_id platform_id = NULL;
    cl_context       context = NULL;       // compute context
    cl_command_queue commands = NULL;      // compute command queue
    cl_program       program = NULL;       // compute program
    cl_kernel        ko_vsqr = NULL;       // compute kernel

    cl_uint num_platforms;
    cl_uint num_devices;
    cl_int ret;


    ret = clGetPlatformIDs(1, &platform_id, &num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_devices);

    devices_time = wtime();
    

    const float update = learning_rate / num_of_samples;

    // Data holders for training/testing.
        // CPU variables
        float *training, *test, *labels_train, *labels_test, *weights;

        // GPU variables
        cl_mem d_training, d_test, d_labels_train, d_labels_test, d_weights. d_gradient;

    // Metrics variables (GPU)    
    cl_mem d_test_accuracy, d_precision, d_recall, d_fone;
    // (CPU)
    float test_accuracy, precision, recall, fone;

    // CPU variables to hold training metrics from each epoch
    float *accuracies, *losses;

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

    weights = (float *)malloc(NUM_PIXELS*sizeof(float));
    
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
                
        } else {
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

    setup_time = wtime(); // clock();


    FILE *file;
    char fileName[] = "./regressor_01.c";
    char *KernelSource;
    size_t kernel_src_size;

    /* Load the source code containing the kernel*/
    file = fopen(fileName, "rb");
    if (!file) {
    fprintf(stderr, "Failed to load kernel.\n");
    exit(1);
    }
    
    fseek(file, 0L, SEEK_END);
    int sz = ftell(file);
    rewind(file);
    KernelSource = (char*)malloc(sz + 1);
    KernelSource[sz] = '\0';
    kernel_src_size = fread(KernelSource, sizeof(char), sz, file); // returns the total number of elements successfully read
    fclose(file);

        output_device_info(device_id);
        
        // Create a compute context
        context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
        if (ret != CL_SUCCESS)
        {
            size_t len;
            char buffer[4350];
            printf("ERRO: create context %s\n", err);
        
            return EXIT_FAILURE;
        }
        
        commands = clCreateCommandQueue(context, device_id, 0, &ret);
        
        if (ret != CL_SUCCESS)
        {
            size_t len;
            char buffer[4350];
            printf("ERRO: create command queue %s\n", err);
        
            return EXIT_FAILURE;
        }


        // Create the compute program from the source buffer
        program = clCreateProgramWithSource(context, 1, (const char **) &KernelSource, (const size_t *)&kernel_src_size, &ret);
        if (ret != CL_SUCCESS)
        {
            size_t len;
            char buffer[4350];
            printf("ERRO: create program with source %s\n", ret);
        
            return EXIT_FAILURE;
        }

        // Build the program
        err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
        
        if (err != CL_SUCCESS)
        {
            size_t len;
            char buffer[16384];
            printf("ERRO Build Program: %d\n", err);
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);       
            printf("%s\n", buffer);
            //return EXIT_FAILURE;
        }


        // Create the compute kernel from the program
        ko_vsqr = clCreateKernel(program, "vsqr", &err);

        // Create the input and output buffers for kernel function
        d_training  = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * num_of_samples * NUM_PIXELS, NULL, &err);

        d_test  = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * num_of_samples * NUM_PIXELS, NULL, &err);

        d_labels_train = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * num_of_samples, NULL, &err);

        d_labels_test = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * TEST_SAMPLES, NULL, &err);

        d_weights = clCreateBuffer(context,  CL_MEM_READ_WRITE,  sizeof(float) * NUM_PIXELS, NULL, &err);

        d_test_accuracy = clCreateBuffer(context,  CL_MEM_WRITE_ONLY,  sizeof(float), NULL, &err);

        d_precision = clCreateBuffer(context,  CL_MEM_WRITE_ONLY,  sizeof(float), NULL, &err);

        d_recall = clCreateBuffer(context,  CL_MEM_WRITE_ONLY,  sizeof(float), NULL, &err);

        d_fone = clCreateBuffer(context,  CL_MEM_WRITE_ONLY,  sizeof(float), NULL, &err);

        d_gradient = clCreateBuffer(context, CL_MEM_READ_WRITE, NUM_PIXELS * sizeof(float), NULL, &err);



        // Write input arguments into compute device memory
        err = clEnqueueWriteBuffer(commands, d_training, CL_TRUE, 0, sizeof(float) * num_of_samples * NUM_PIXELS, training, 0, NULL, NULL);

        err = clEnqueueWriteBuffer(commands, d_test, CL_TRUE, 0, sizeof(float) * num_of_samples * NUM_PIXELS, test, 0, NULL, NULL);

        err = clEnqueueWriteBuffer(commands, d_labels_train, CL_TRUE, 0, sizeof(float) * num_of_samples, labels_train, 0, NULL, NULL);

        err = clEnqueueWriteBuffer(commands, d_labels_test, CL_TRUE, 0, sizeof(float) * TEST_SAMPLES, labels_test, 0, NULL, NULL);

        err = clEnqueueWriteBuffer(commands, d_weights, CL_TRUE, 0, sizeof(float) * NUM_PIXELS, weights, 0, NULL, NULL);

        err = clEnqueueReadBuffer(commands, d_test_accuracy, CL_TRUE, 0, sizeof(float), &test_accuracy, 0, NULL, NULL);        

        err = clEnqueueReadBuffer(commands, d_precision, CL_TRUE, 0, sizeof(float), &precision, 0, NULL, NULL);

        err = clEnqueueReadBuffer(commands, d_recall, CL_TRUE, 0, sizeof(float), &recall, 0, NULL, NULL);

        err = clEnqueueReadBuffer(commands, d_fone, CL_TRUE, 0, sizeof(float), &fone, 0, NULL, NULL);

        // Set the arguments to our compute kernel
        cl_uint d_i = 0;
        err  = clSetKernelArg(ko_vsqr, d_i++, sizeof(cl_mem), &d_training);
        err |= clSetKernelArg(ko_vsqr, d_i++, sizeof(cl_mem), &d_test);
        err |= clSetKernelArg(ko_vsqr, d_i++, sizeof(cl_mem), &d_labels_train);
        err |= clSetKernelArg(ko_vsqr, d_i++, sizeof(cl_mem), &d_labels_test);
        err |= clSetKernelArg(ko_vsqr, d_i++, sizeof(cl_mem), &d_weights);
        err |= clSetKernelArg(ko_vsqr, d_i++, sizeof(cl_mem), &d_gradient);
        err |= clSetKernelArg(ko_vsqr, d_i++, sizeof(cl_mem), &d_test_accuracy);
        err |= clSetKernelArg(ko_vsqr, d_i++, sizeof(cl_mem), &d_precision);
        err |= clSetKernelArg(ko_vsqr, d_i++, sizeof(cl_mem), &d_recall);
        err |= clSetKernelArg(ko_vsqr, d_i++, sizeof(cl_mem), &d_fone);
        err |= clSetKernelArg(ko_vsqr, d_i++, sizeof(int), &num_test_samples);
        err |= clSetKernelArg(ko_vsqr, d_i++, sizeof(int), &num_of_epochs);
        err |= clSetKernelArg(ko_vsqr, d_i++, sizeof(int), &num_of_samples);
        
        rtime = wtime();
        
        // Execute the kernel over the entire range of our 1d input data set
        // letting the OpenCL runtime choose the work-group size
        global = num_of_samples * NUM_PIXELS;
        err = clEnqueueNDRangeKernel(commands, ko_vsqr, 1, NULL, &global, NULL, 0, NULL, NULL);
        
        // Wait for the commands to complete before stopping the timer
        err = clFinish(commands);
        

        rtime = wtime() - rtime;
        printf("\nThe kernel ran in %lf seconds\n",rtime);
        
        // Read back the results from the compute device
        err = clEnqueueReadBuffer(commands, d_test_accuracy, CL_TRUE, 0, sizeof(float), &test_accuracy, 0, NULL, NULL );
        if (err != CL_SUCCESS)
        {
            exit(1);
        }

        err = clEnqueueReadBuffer(commands, d_precision, CL_TRUE, 0, sizeof(float), &precision, 0, NULL, NULL );
        if (err != CL_SUCCESS)
        {
            exit(1);
        }

        err = clEnqueueReadBuffer(commands, d_recall, CL_TRUE, 0, sizeof(float), &recall, 0, NULL, NULL );
        if (err != CL_SUCCESS)
        {
            exit(1);
        }

        err = clEnqueueReadBuffer(commands, d_fone, CL_TRUE, 0, sizeof(float), &fone, 0, NULL, NULL );
        if (err != CL_SUCCESS)
        {
            exit(1);
        }

        // cleanup then shutdown
        clReleaseMemObject(d_training);
        clReleaseMemObject(d_test);
        clReleaseMemObject(d_labels_train);
        clReleaseMemObject(d_labels_test);
        clReleaseMemObject(d_test_accuracy);
        clReleaseMemObject(d_precision);
        clReleaseMemObject(d_recall);
        clReleaseMemObject(d_fone);
        clReleaseProgram(program);
        clReleaseKernel(ko_vsqr);
        clReleaseCommandQueue(commands);
        clReleaseContext(context);

    end_time = wtime(); // clock();

    printf("oxe\n");

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
