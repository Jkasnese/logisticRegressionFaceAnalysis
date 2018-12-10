// Extracted from:
// https://mmlind.github.io/Using_Logistic_Regression_to_solve_MNIST/

/**
 * @file loopopt.h
 * @author Manuella Vieira e Guilherme Lopes
 * @date 2/11/2018

scp to umbu.uefs.br (172.16.112.7)
mpicc -o loopmpi.out loopmpi.c -std=c99 -fopenmp -lm     
nohup mpirun -np 3 -machinefile machines.txt loopmpi.out 50 4487 3 1   - (epoch) (samples) (number of processes) (number of threads (OpenMP)).

 */
#define _GNU_SOURCE

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<omp.h>
//#include<sched.h>
#include<mpi.h>

#define TOTAL_SAMPLES 5481
#define TRAINING_SAMPLES 4487
#define TEST_SAMPLES 994
#define NUM_PIXELS 16385 // Num pixels + 1 (bias)

// #define NODES 3

const float learning_rate = 0.01; /**< Constant that holds the learning rate */

int main(int argc, char *argv[]){

    int rank;
    
    MPI_Init(&argc, &argv); 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double start_time, malloc_time, setup_time, gen_weights_time, bcast_time, divide_imgs_time, scatter_time; // clock();
    double start_reduce_time, end_reduce_time, training_time, test_time, files_time;

    start_time = MPI_Wtime();
    
    int num_of_epochs = atoi(argv[1]); 
    int num_of_samples = atoi(argv[2]);
    int num_of_nodes = atoi(argv[3]);
    int threads = atoi(argv[4]);
    
    
    /**
     * - Defines the number of threads to be used by OpenMP for parallelization based on argument provided by the user
    */
    omp_set_num_threads(threads);
    
    // Training/testing variables. Parallel variables.
    float *training, *test;
    float labels_train[TRAINING_SAMPLES], labels_test[TEST_SAMPLES], pixel;
        // Metrics holders
    float accuracies[num_of_epochs], losses[num_of_epochs];
    float test_accuracy = 0, precision = 0, recall = 0, fone = 0;
    
    training = (float *)malloc((TRAINING_SAMPLES * NUM_PIXELS)*sizeof(float));
    test = (float *)malloc((TEST_SAMPLES * NUM_PIXELS)*sizeof(float));
    
    // Weights
        /** - Generates weight matrix */
    float* weights;
    weights = (float *)malloc(NUM_PIXELS*sizeof(float));

    // Generate array to hold hypothesis results:
    float* hypothesis;
    hypothesis = (float *) malloc (TRAINING_SAMPLES*sizeof(float));

    float* gradient;
    gradient = (float *) malloc (NUM_PIXELS*sizeof(float));

    float* global_gradient;
    global_gradient = (float *) malloc (NUM_PIXELS*sizeof(float));

    const float update = learning_rate/TRAINING_SAMPLES;

    float temp = 0;
    float aux = 0;
    int right_answers = 0;
    float loss = 0;
    float global_loss = 0;
    int *displs, *sendcounts;
    int remainder = 0, count = 0;
    int num_of_training_imgs; 

    // clock();
    malloc_time = MPI_Wtime();

    if (rank == 0){
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

        setup_time = MPI_Wtime(); // clock();

        /** - Parallelizes the loop for initializing weight values */
        #pragma omp parallel for
        for (int i=0; i<NUM_PIXELS; i++){
            weights[i] =  ( (rand() % 100) / 146.0) - 0.35; //>> 2 fica quanto mais rápido?
            //if(i % 10 == 0)
              //   printf("i = %d thread num = %d core = %d\n", i, omp_get_thread_num(), sched_getcpu());
        }
    }

    gen_weights_time = MPI_Wtime(); // clock();

    // BROADCAST dos pesos - MPI
    MPI_Bcast(weights, NUM_PIXELS, MPI_FLOAT, 0, MPI_COMM_WORLD);

    bcast_time = MPI_Wtime();    // clock();

    aux = num_of_samples/num_of_nodes;
 

    if(rank == 0){

        displs = (int *) malloc(num_of_nodes*sizeof(int));
        sendcounts = (int *) malloc(num_of_nodes*sizeof(int));

        // Dividindo a quantidade de imagens. MPI. num_of_samples % nodes
        num_of_training_imgs = num_of_samples/num_of_nodes + (num_of_samples % num_of_nodes);
        displs[0] = num_of_samples;
        sendcounts[0] = aux;
        temp = num_of_samples;
        for(int i=1; i<num_of_nodes; i++){
            sendcounts[i] = aux;
            temp += aux;
            displs[i] = temp;
        }
    } else {
        num_of_training_imgs = aux;
    }

    divide_imgs_time = MPI_Wtime(); // clock();
    
    MPI_Scatterv(training, sendcounts, displs, MPI_FLOAT, training, aux, MPI_FLOAT, 0, MPI_COMM_WORLD);

    scatter_time = MPI_Wtime(); // clock();
         
    // BEGINING OF TRAINING EPOCHS
    int r_numpixels;
    for (int epochs=0; epochs<num_of_epochs; epochs++){
        // Zeroing epoch stats
        right_answers = 0;
        loss = 0;

        // Zeroing gradients from previous epoch
        for (int i = 0; i < NUM_PIXELS; ++i)
        {
            gradient[i] = 0;
        }

        /** - Parallelizes generation of hypothesis values for each sample */
        // MPI - cada nó tem um número de amostras diferentes. 
        #pragma omp parallel for private(temp, aux) reduction(+:loss) 
        for (long r=0; r<num_of_training_imgs; r++){
            r_numpixels = r*NUM_PIXELS;
            temp = 0;
            for (long x=0; x<NUM_PIXELS; x++){
                temp += *(training + (r_numpixels)+x) * *(weights + x);
            }
            /** - Calculates logistic hypothesis */
            temp = 1 / (1 + (exp( -1.0 * temp)) );

            
            /** - Computes loss function */
            aux = labels_train[r]*log(temp) + (1 - labels_train[r])*log(1-temp);
            loss += aux; // Acelera se trocar por if/else dos labels?

            /** - Computes the difference between label and hypothesis */
            temp = labels_train[r] - temp;

            /** - Computes accuracy on training set */
            if (temp < 0){
                aux = aux*-1; //Há como acelerar simplesmente manipulando os bits?
            }
            if (aux < 0.5){
                right_answers++;
            }

            /** - Computes current gradient */
            r_numpixels = r*NUM_PIXELS;
            for (long x=0; x<NUM_PIXELS; x++){
                gradient[x] += training[r_numpixels + x] * temp;
            }
        }

        start_reduce_time = MPI_Wtime(); // clock();

        // 2 modos de fazer: reduce + broadcast dos pesos ou reduce + broadcast dos gradientes (allreduce)
        // MPI - Reduce loss & gradient
        MPI_Reduce(&loss, &global_loss, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

        MPI_Allreduce(gradient, global_gradient, NUM_PIXELS, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        end_reduce_time = MPI_Wtime(); // clock();

        /** - Updates weights */
        for (int i=0; i<NUM_PIXELS; i++){
            weights[i] += update * global_gradient[i];
        }

        /** - Saves epoch metrics to be plotted later */
        if (rank == 0){
            accuracies[epochs] = ((float) right_answers) / TRAINING_SAMPLES;
            losses[epochs] = global_loss;
        }

    }

    training_time = MPI_Wtime();    // clock();
    if (rank==0){

        // CALCULATE TEST METRICS
        // Zeroing variables to hold metrics stats:
        right_answers = 0;
        int fp = 0, tp = 0, tn = 0, fn = 0;


        /** - Generate hypothesis values for the test set */
        #pragma omp parallel for private(temp)
        for (long r=0; r<TEST_SAMPLES; r++){
            r_numpixels = r*NUM_PIXELS;
            temp = 0;
            for (long x=0; x<NUM_PIXELS; x++){
                temp += *(test + r_numpixels+x) * weights[x];
            }
            
            // Calculate logistic hypothesis
            temp = 1 / (1 + (exp( -1.0 * temp)) );

            // Compute the difference between label and hypothesis &
            //  accuracy on training set &
            //  loss function &
            //  metrics (accuracy, precision, recall, f1)
            if (labels_test[r] == 1.0){
                if (temp < 0.5){
                    // FP
                    fp++;
                } else {
                    // TP
                    tp++;
                }
            } else {
                if (temp < 0.5){
                    // TN
                    tn++;
                } else {
                    // FN
                    fn++;
                }
            }
        }

        test_accuracy = ((float) (tp + tn))/ TEST_SAMPLES;
        precision = ((float) tp) / (tp+fp);
        recall = ((float) tp) / (tp + fn);
        fone = 2*((precision*recall) / (precision + recall));

        test_time = MPI_Wtime(); // clock();

        printf("%s %f\n%s %f\n%s %f\n%s %f\n", "accuracy ", test_accuracy, "precision ", precision, "recall ", recall, "f1 ", fone);


        /** - Writes metrics (accuracy, loss, precision, recall and F1 score) to files */ 
        FILE* facc = fopen("training_acc.txt", "w");
        FILE* floss = fopen("loss.txt", "w");
        FILE* ftest = fopen("test_metrics.txt", "w");

       for (int i = 0; i < num_of_epochs; ++i)
        {
            fprintf(facc, "%f\n", accuracies[i]);
            fprintf(floss, "%f\n", losses[i]);
        }

        fprintf(ftest, "%s %f\n%s %f\n%s %f\n%s %f\n", "accuracy ", test_accuracy, "precision ", precision, "recall ", recall, "f1 ", fone);

        fclose(facc);
        fclose(floss);
        fclose(ftest);
    }

    files_time = MPI_Wtime(); // clock();

    double start_time, malloc_time, setup_time, gen_weights_time, bcast_time, divide_imgs_time, scatter_time; // clock();
    double start_reduce_time, end_reduce_time, training_time, test_time, files_time;

    FILE* execution_times = fopen("execution_times", "w");

    fprintf(execution_times, "%s\t%0.9f\n", "memory_allocation_time: ", (malloc_time - start_time) );
    fprintf(execution_times, "%s\t%0.9f\n", "Setup Serial time: ", (setup_time - start_time) );
    fprintf(execution_times, "%s\t%0.9f\n", "gen_weights_time: ", (gen_weights_time - setup_time) );
    fprintf(execution_times, "%s\t%0.9f\n", "bcast_time: ", (bcast_time - gen_weights_time) );
    fprintf(execution_times, "%s\t%0.9f\n", "divide_imgs_time: ", (divide_imgs_time - bcast_time) );
    fprintf(execution_times, "%s\t%0.9f\n", "scatter_time: ", (scatter_time - divide_imgs_time) );
    fprintf(execution_times, "%s\t%0.9f\n", "reduce+allreduce_time: ", (end_reduce_time - start_reduce_time) );
    fprintf(execution_times, "%s\t%0.9f\n", "Training time: ", (scatter_time - training_time) );
    fprintf(execution_times, "%s\t%0.9f\n", "Test time: ", (test_time - training_time) );
    fprintf(execution_times, "%s\t%0.9f\n", "files_time ", (files_time - test_time) );
    fprintf(execution_times, "%s\t%0.9f\n", "Total time: ", (files_time - start_time) );

    MPI_Finalize();
}
