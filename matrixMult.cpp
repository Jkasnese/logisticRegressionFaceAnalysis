// Extracted from:
// https://mmlind.github.io/Using_Logistic_Regression_to_solve_MNIST/

#define NUM_PIXELS 2305 // Num pixels + 1 (bias)
#define NUM_EXAMPLES 35900

void multiply_matrices(double *matrix1, double *matrix2, double *result){ // quanto o dgemm é mais rápido?
    
    // Inicializar a matriz de resultados.
    // Não precisa, porque podemos armazenar o valor num registrador e salvar o resultado
    // direto na matriz.
    float temp = 0;
    
    for (long r=0; r<NUM_EXAMPLES; r++){        
        for (long x=0; x<NUM_PIXELS; x++){
            temp += *(matrix1 + (r*NUM_PIXELS)+x) * *(matrix2 + x);
        }
        *(result + r) = temp;
    }
}

void sigmoid(double *outputs){
    double *val = outputs;
    for (int i=0; i<NUM_EXAMPLES; i++) {
        val = val[i];
        *val = 1 / (1 + (exp((double)-*val)) );
    }
}


// Generate weight matrix
float* weights;

weights = (float *)malloc(NUM_PIXELS*sizeof(float));

for (int i=0; i<NUM_PIXELS; i++){
    weight[i] = (float) rand(%2)/2.0; //>> 2 fica quanto mais rápido?
}

// Generate array to hold hypotesis results:
float* hypotesis;
hypotesis = (float *) malloc (NUM_EXAMPLES*sizeof(float));

multiply_matrices(training, weights, hypotesis);
sigmoid(hypotesis);










