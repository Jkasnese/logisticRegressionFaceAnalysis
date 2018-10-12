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

int main(){

    char filename[] = "fer2013.csv";
    int buffer_size = 10000, emotion_6 = 0, emotion_4 = 0;
    char *charbuffer, *emotion, *usage, *char_pixels, *temp_pixels;
    float *training, *test;
    float labels_train[TRAINING_SAMPLES], labels_test[TEST_SAMPLES], pixel;

    // Metrics holders
    float accuracies[NUM_EPOCHS], losses[NUM_EPOCHS];
    float test_accuracy, precision, recall, fone;

	FILE* FER_images = fopen(filename, "r");
    int i_train = 0, i_test = 0, i = 0;
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

            // Se colocar isso dentro do if abaixo, em quantas vezes acelera?
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

                for (int j = 0; j < 2304; j++){
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
                    i_train++;
                }
                else{
                    i_test++;
                }
          }
    }

    // Arquivo lido. Fechar arquivo.
    fclose(FER_images);

    // Adding bias
    training[NUM_PIXELS-1] = 1;
    test[NUM_PIXELS-1] = 1;

    // COMEÇO DO TREINO - BEGINNING OF TRAINING STAGE:

    // Generate weight matrix
    float* weights;

    weights = (float *)malloc(NUM_PIXELS*sizeof(float));

    for (int i=0; i<NUM_PIXELS; i++){
        weights[i] =  ( (rand() % 100) / 143.0) - 0.35; //>> 2 fica quanto mais rápido?
    }

    // Generate array to hold hypotesis results:
    float* hypotesis;
    hypotesis = (float *) malloc (TRAINING_SAMPLES*sizeof(float));

    float* gradient;
    gradient = (float *) malloc (NUM_PIXELS*sizeof(float));

    const float update = learning_rate/TRAINING_SAMPLES;

    float temp = 0;
    int right_answers = 0;
    float* dif;
    float* val;
    float loss = 0;

    // BEGINING OF TRAINING EPOCHS

    for (int epochs=0; epochs<NUM_EPOCHS; epochs++){
        
        // Generate hypotesis values
        for (long r=0; r<TRAINING_SAMPLES; r++){        
            for (long x=0; x<NUM_PIXELS; x++){
                temp += *(training + (r*NUM_PIXELS)+x) * *(weights + x);
            }
            *(hypotesis + r) = temp;
            temp = 0;
        }

        // Calculate logistic hypotesis
        val = hypotesis;
        for (int i=0; i<TRAINING_SAMPLES; i++) {
            
            *val = 1 / (1 + (exp( -1.0 * *val)) );
           // printf("%f\n", *val );
            val++;
        }

        // Compute the difference between label and hypotesis & 
        //  accuracy on training set &
        //  loss function &
        //  metrics (accuracy, precision, recall, f1)
        dif = hypotesis;
        right_answers = 0;
        loss = 0;
        for (int i = 0; i < TRAINING_SAMPLES; ++i) {
            temp = labels_train[i] - dif[i];
            loss -= labels_train[i]*log(dif[i]) + (1 - labels_train[i])*log(1-dif[i]); // Acelera se trocar por if/else dos labels?
            //printf("%f\n", temp);
            dif[i] = temp;
            if (temp < 0){
                temp = temp*-1; //Há como acelerar simplesmente manipulando os bits?
            }
            if (temp < 0.5){
                right_answers++;
            }
        }

        // Saving metrics to be ploted later
        accuracies[epochs] = ((float) right_answers) / TRAINING_SAMPLES;
        losses[epochs] = loss;
//        printf("%f\n", loss);
        //printf("%f\n", ((float) right_answers) / TRAINING_SAMPLES*100);

        // Compute the gradient
        temp = 0;
        for (long r=0; r<NUM_PIXELS; r++){        
            for (long x=0; x<TRAINING_SAMPLES; x++){
                //printf("%f\n", temp);
                temp += *(training + (NUM_PIXELS*x)+r) * hypotesis[x];
            }
            
            *(gradient + r) = temp;
            temp = 0;
        }
        
        // Update weights
        for (int i=0; i<NUM_PIXELS; i++){
            weights[i] += update * gradient[i];
            //printf("%f\n", weights[i]);
        }
    }

    // CALCULATE TEST METRICS
    // Generate hypotesis values
    for (long r=0; r<TEST_SAMPLES; r++){        
        for (long x=0; x<NUM_PIXELS; x++){
            temp += *(test + (r*NUM_PIXELS)+x) * *(weights + x);
        }
        *(hypotesis + r) = temp;
        temp = 0;
    }

    // Calculate logistic hypotesis
    val = hypotesis;
    for (int i=0; i<TEST_SAMPLES; i++) {
        
        *val = 1 / (1 + (exp( -1.0 * *val)) );
       // printf("%f\n", *val );
        val++;
    }

    // Compute the difference between label and hypotesis & 
    //  accuracy on training set &
    //  loss function &
    //  metrics (accuracy, precision, recall, f1)
    dif = hypotesis;
    right_answers = 0;
    int fp = 0, tp = 0, tn = 0, fn = 0;
    for (int i = 0; i < TEST_SAMPLES; ++i) {
        //temp = labels_train[i] - dif[i];
        // dif[i] = temp;
        if (labels_train[i] == 1.0){
            if (dif[i] < 0.5){
                // FP
                fp++;
            } else {
                // TP
                tp++;
            }
        } else {
            if (dif[i] < 0.5){
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

    // Write data to files
    FILE* facc = fopen("training_acc.txt", "w");
    FILE* floss = fopen("loss.txt", "w");
    FILE* ftest = fopen("test_metrics.txt", "w");

    for (int i = 0; i < NUM_EPOCHS; ++i)
    {
        fprintf(facc, "%f\n", accuracies[i]);
        fprintf(floss, "%f\n", losses[i]);
    }

    fprintf(ftest, "%s %f\n%s %f\n%s %f\n%s %f\n", "accuracy ", test_accuracy, "precision ", precision, "recall ", recall, "f1 ", fone);

    fclose(facc);
    fclose(floss);
    fclose(ftest);
}
