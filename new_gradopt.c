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
#define NUM_EPOCHS 50

const float learning_rate = 0.01;

int main(int argc, char *argv[]){

    printf("rodando\n");
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

    // COMEÇO DO TREINO - BEGINNING OF TRAINING STAGE:

    // Generate weight matrix
     float* weights;

    weights = (float *)malloc(NUM_PIXELS*sizeof(float));

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
    float* dif;
    float* val;
    float aux;
    float loss = 0;

    // BEGINING OF TRAINING EPOCHS
    int r_numpixels;
    for (int epochs=0; epochs<num_of_epochs; epochs++){

        // Generate hypothesis values
        for (long r=0; r<TRAINING_SAMPLES; r++){
            r_numpixels = r*NUM_PIXELS;
            for (long x=0; x<NUM_PIXELS; x++){
                temp += *(training + (r_numpixels)+x) * *(weights + x);
            }
            *(hypothesis + r) = temp;
            temp = 0;
        }

        // Calculate logistic hypothesis
        val = hypothesis;
        for (int i=0; i<TRAINING_SAMPLES; i++) {

            *val = 1 / (1 + (exp( -1.0 * *val)) );
           // printf("%f\n", *val );
            val++;
        }

        // Compute the difference between label and hypothesis &
        //  accuracy on training set &
        //  loss function &
        //  metrics (accuracy, precision, recall, f1)
        dif = hypothesis;
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

        for (int i = 0; i < NUM_PIXELS; ++i)
        {
            gradient[i] = 0;
        }
        // Compute the gradient
        for (long r=0; r<TRAINING_SAMPLES; r++){
            aux = hypothesis[r];
            r_numpixels = r*NUM_PIXELS;
            for (long x=0; x<NUM_PIXELS; x++){
                //printf("%f\n", temp);
                gradient[x] += training[r_numpixels + x] * aux;
            }
        }

        // Update weights
        for (int i=0; i<NUM_PIXELS; i++){
            weights[i] += update * gradient[i];

            //printf("%f\n", weights[i]);
        }
    }

    // CALCULATE TEST METRICS
    // Generate hypothesis values
    for (long r=0; r<TEST_SAMPLES; r++){
        r_numpixels = r*NUM_PIXELS;
        for (long x=0; x<NUM_PIXELS; x++){
            temp += *(test + r_numpixels+x) * weights[x];
        }
        *(hypothesis + r) = temp;
        temp = 0;
    }

    // Calculate logistic hypothesis
    val = hypothesis;
    for (int i=0; i<TEST_SAMPLES; i++) {

        *val = 1 / (1 + (exp( -1.0 * *val)) );
       // printf("%f\n", *val );
        val++;
    }

    // Compute the difference between label and hypothesis &
    //  accuracy on training set &
    //  loss function &
    //  metrics (accuracy, precision, recall, f1)
    dif = hypothesis;
    right_answers = 0;
    int fp = 0, tp = 0, tn = 0, fn = 0;
    for (int i = 0; i < TEST_SAMPLES; ++i) {
        //temp = labels_train[i] - dif[i];
        // dif[i] = temp;
        if (labels_test[i] == 1.0){
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
