// Extracted from:
// https://mmlind.github.io/Using_Logistic_Regression_to_solve_MNIST/

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>

#define TOTAL_SAMPLES 12275
#define TRAINING_SAMPLES 9795
#define TEST_SAMPLES 2480
#define NUM_PIXELS 2305 // Num pixels + 1 (bias)
#define NUM_EPOCHS 500

const float learning_rate = 0.01;

int main(int argc, char *argv[]){

    int num_of_epochs = atoi(argv[1]);


    char filename[] = "fer2013.csv";
    int buffer_size = 10000, emotion_6 = 0, emotion_4 = 0;
    char *charbuffer, *emotion, *usage, *char_pixels, *temp_pixels;
    float *training, *test;
    float labels_train[TRAINING_SAMPLES], labels_test[TEST_SAMPLES], pixel;

    // Metrics holders
    float accuracies[num_of_epochs], losses[num_of_epochs];
    float test_accuracy, precision, recall, fone;

    FILE* FER_images = fopen(filename, "r");
    int i_train = 0, i_test = 0, i = 0, j = 0;
    int offset = 0;
    int is_training; // 1 == training, 0 == test

    charbuffer = (char *)malloc(buffer_size*sizeof(char));
    temp_pixels = (char *)malloc(2*sizeof(float));
    training = (float *)malloc((TRAINING_SAMPLES * 2305)*sizeof(float));
    test = (float *)malloc((TEST_SAMPLES * 2305)*sizeof(float));


    // Lendo todo o arquivo para treino
    while(fgets(charbuffer, buffer_size, FER_images) != NULL) {
            emotion = strtok(charbuffer, ",");
            char_pixels = strtok(NULL, ",");
            usage = strtok(NULL, ",");

        
            if(strcmp(usage, "Training\n") == 0){
                is_training = 1;
            }
            else
                is_training = 0;

            if (strcmp(emotion, "6") == 0 || strcmp(emotion, "4") == 0){
                if (strcmp(emotion, "6") == 0){
                    if(is_training == 1){
                        labels_train[i_train] = 1;
                        emotion_6++;
                    }
                    else{
                        labels_test[i_test] = 1;
                    }

                }
                else{
                    if(is_training == 1){
                        labels_train[i_train] = 0;
                        emotion_4++;
                    }
                    else{
                        labels_test[i_test] = 0;
                    }
                }

                temp_pixels = strtok(char_pixels, " ");

                for (j = 0; j < 2304; j++){
                    pixel = atof(temp_pixels);

                    if(is_training == 1){
                        offset = i_train*2305 + j;
                        training[offset] = pixel/255.0;
                    }
                    else{
                        offset = i_test*2305 + j;
                        test[offset] = pixel/255.0;
                    }

                    temp_pixels = strtok(NULL, " ");
                }

                if(is_training == 1){

                    training[i_train*NUM_PIXELS + j] = 1;
                    i_train++;
                    
                }
                else{
                    test[i_test*NUM_PIXELS + j] = 1;
                    i_test++;
                }
          }
    }


    // Arquivo lido. Fechar arquivo.
    fclose(FER_images);

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
        //  #pragma omp parallel for private(temp) reduction(-:loss)
        for (long r=0; r<TRAINING_SAMPLES; r++){
            r_numpixels = r*NUM_PIXELS;
            temp = 0;
            for (long x=0; x<NUM_PIXELS; x++){
                temp += *(training + (r_numpixels)+x) * *(weights + x);
            }

            // Calculate logistic hypothesis
            hypothesis[r] = 1 / (1 + (exp( -1.0 * temp)) );
            
            // Compute the difference between label and hypothesis &
            //  accuracy on training set & loss function
            temp = labels_train[r]*log(hypothesis[r]) + (1 - labels_train[r])*log(1-hypothesis[r]);
            // Precisa de semáforo em loss. Ou cada um tem sua loss e no final soma as losses.
            printf("%f\n", temp);
            loss -= temp; // Acelera se trocar por if/else dos labels?
            hypothesis[r] = temp;

            if (temp < 0){
                temp = temp*-1; //Há como acelerar simplesmente manipulando os bits?
            }
            if (temp < 0.5){
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

            //printf("%f\n", weights[i]);
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
