// Extracted from:
// https://mmlind.github.io/Using_Logistic_Regression_to_solve_MNIST/

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>

#define TOTAL_SAMPLES 5481
#define TRAINING_SAMPLES 4487
#define TEST_SAMPLES 994
#define NUM_PIXELS 16385 // Num pixels + 1 (bias)

const float learning_rate = 0.01;

int main(int argc, char *argv[]){

    int num_of_epochs = atoi(argv[1]);

    char train_filename[] = "fold_training_out.csv", test_filename[] = "fold_test_out.csv";
    int buffer_size = 100000, gender_female = 0, gender_male = 0;
    char *charbuffer, *gender, *usage, *char_pixels, *temp_pixels;
    float *training, *test;
    float labels_train[TRAINING_SAMPLES], labels_test[TEST_SAMPLES], pixel;

    // Metrics holders
    float accuracies[num_of_epochs], losses[num_of_epochs];
    float test_accuracy = 0, precision = 0, recall = 0, fone = 0;

    FILE* train_images = fopen(train_filename, "r");
    FILE* test_images = fopen(test_filename, "r");
    int i_train = 0, i_test = 0, i = 0, j = 0;
    int offset = 0;
    int is_training; // 1 == training, 0 == test

    charbuffer = (char *)malloc(buffer_size*sizeof(char));
    temp_pixels = (char *)malloc(2*sizeof(float));
    training = (float *)malloc((TRAINING_SAMPLES * NUM_PIXELS)*sizeof(float));
    test = (float *)malloc((TEST_SAMPLES * NUM_PIXELS)*sizeof(float));

    // Lendo todo o arquivo para teste
    while(fgets(charbuffer, buffer_size, train_images) != NULL) {
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
    
    // Arquivos lidos. Fechar arquivos.
    fclose(train_images);
    fclose(test_images);


    // Adding bias

    // COMEÇO DO TREINO - BEGINNING OF TRAINING STAGE:

    // Generate weight matrix
    float* weights;

    weights = (float *)malloc(NUM_PIXELS*sizeof(float));


    // Paraleliza
    for (int i=0; i<NUM_PIXELS; i++){
        weights[i] =  ( (rand() % 100) / 146.0) - 0.35; //>> 2 fica quanto mais rápido?
    }

    // Generate array to hold hypothesis results:
    float* hypothesis;
    hypothesis = (float *) malloc (TRAINING_SAMPLES*sizeof(float));

    float* gradient;
    gradient = (float *) malloc (NUM_PIXELS*sizeof(float));

    const float update = learning_rate/TRAINING_SAMPLES;

    float temp = 0;
    float aux = 0;
    int right_answers = 0;
    float loss = 0;

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

        // Generate hypothesis values for each sample
        #pragma omp parallel for private(temp, aux) reduction(-:loss)
        for (long r=0; r<TRAINING_SAMPLES; r++){
            r_numpixels = r*NUM_PIXELS;
            temp = 0;
            for (long x=0; x<NUM_PIXELS; x++){
                temp += *(training + (r_numpixels)+x) * *(weights + x);
            }

            // Calculate logistic hypothesis
            temp = 1 / (1 + (exp( -1.0 * temp)) );

            //printf("%f\n", temp);
            
            // Compute loss function
            aux = labels_train[r]*log(temp) + (1 - labels_train[r])*log(1-temp);
            // Precisa de semáforo em loss. Ou cada um tem sua loss e no final soma as losses.
            loss -= aux; // Acelera se trocar por if/else dos labels?

            // Compute the difference between label and hypothesis
            temp = labels_train[r] - temp;

            // Compute accuracy on training set
            if (temp < 0){
                aux = aux*-1; //Há como acelerar simplesmente manipulando os bits?
            }
            if (aux < 0.5){
                right_answers++;
            }

            // Compute current gradient
            // Precisa de semáforo em gradiente
            r_numpixels = r*NUM_PIXELS;
            for (long x=0; x<NUM_PIXELS; x++){
                gradient[x] += training[r_numpixels + x] * temp;
            }
        }

        // Update weights
        // Paraleliza
        for (int i=0; i<NUM_PIXELS; i++){
            weights[i] += update * gradient[i];

           // printf("%f\n", weights[i]);
        }

        // Saving epoch metrics to be ploted later
        accuracies[epochs] = ((float) right_answers) / TRAINING_SAMPLES;
        losses[epochs] = loss;
    }

    // CALCULATE TEST METRICS
    // Zeroing variables to hold metrics stats:
    right_answers = 0;
    int fp = 0, tp = 0, tn = 0, fn = 0;


    // Generate hypothesis values
    // Paraleliza
    for (long r=0; r<TEST_SAMPLES; r++){
        r_numpixels = r*NUM_PIXELS;
        temp = 0;
        for (long x=0; x<NUM_PIXELS; x++){
            temp += *(test + r_numpixels+x) * weights[x];
        }
        *(hypothesis + r) = temp;
        
        // Calculate logistic hypothesis
        hypothesis[r] = 1 / (1 + (exp( -1.0 * temp)) );

        // Compute the difference between label and hypothesis &
        //  accuracy on training set &
        //  loss function &
        //  metrics (accuracy, precision, recall, f1)
        if (labels_test[r] == 1.0){
            if (hypothesis[r] < 0.5){
                // FP
                fp++;
            } else {
                // TP
                tp++;
            }
        } else {
            if (hypothesis[r] < 0.5){
                // TN
                tn++;
            } else {
                // FN
                fn++;
            }
        }
    }

    // Saving metrics to be ploted later
    test_accuracy = ((float) (tp + tn))/ TEST_SAMPLES;
    precision = ((float) tp) / (tp+fp);
    recall = ((float) tp) / (tp + fn);
    fone = 2*((precision*recall) / (precision + recall));
    printf("%s %f\n%s %f\n%s %f\n%s %f\n", "accuracy ", test_accuracy, "precision ", precision, "recall ", recall, "f1 ", fone);


    // Write data to files
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
