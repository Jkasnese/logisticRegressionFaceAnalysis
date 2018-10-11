#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#define NUM_EXAMPLES  35900
#define NUM_PIXELS 2305 // Number of pixels + 1 (bias)

int main(){


	char filename[] = "fer2013.csv";
	int buffer_size = 10000;
    char *charbuffer, *emotion, *usage, *char_pixels, *temp_pixels;
    float* pixels, *training, *test;
    float labels[NUM_EXAMPLES], pixel;

	FILE* FER_images = fopen(filename, "r");
    int i = 0;
    int j = 0;
    int offset = 0;

    charbuffer = (char *)malloc(buffer_size*sizeof(char));
    pixels = (float *)malloc((2305)*sizeof(float));
    temp_pixels = (char *)malloc(2*sizeof(float));
    training = (float *)malloc((NUM_EXAMPLES * 2305)*sizeof(float));

    // Lendo todo o arquivo para treino

    while(fgets(charbuffer, buffer_size, FER_images) != NULL) {
            emotion = strtok(charbuffer, ",");
            char_pixels = strtok(NULL, ",");
            usage = strtok(NULL, ",");

            if (strcmp(emotion, "6") == 0 || strcmp(emotion, "4") == 0){
                if (emotion == "6"){
                    labels[i] = 1;
                } else{
                    labels[i] = 0;
                }

                temp_pixels = strtok(char_pixels, " ");
                pixel = atof(temp_pixels);
                offset = i*2305;
                training[offset] = pixel/255.0;

                for (j = 1; j < 2304; j++){
                    temp_pixels = strtok(NULL, " ");

                    if(temp_pixels != NULL){
                        pixel = atof(temp_pixels);
                        if(strcmp(usage, "Training\n") == 0){
                            offset = i*2305 + j;
                            training[offset] = pixel/255.0;
                        }
                    }
                }
            i++;
          }
    }


fclose(FER_images);




}
