// Extracted from:
// https://mmlind.github.io/Using_Logistic_Regression_to_solve_MNIST/


void multiply_matrices(double *matrix1, long row1, long col1, double *matrix2, long row2, long col2, double *result){
    
    assert(row1>0);
    assert(col1>0);
    assert(col1==row2);
    assert(col2>0);
    
    long num_rows = row1;
    long num_cols = col2;
    long num_x    = col1;
    
    
    // Inicializar a matriz de resultados.
    // Não precisa, porque podemos armazenar o valor num registrador e salvar o resultado
    // direto na matriz.
    for (long i=0; i<row1*col2; i++) *(result+i)=0;
    
    for (long c=0; c<num_cols; c++){
        for (long r=0; r<num_rows; r++){        
            for (long x=0; x<num_x; x++){
                *(result + (r*num_cols) + c) += *(matrix1 + (r*num_x)+x) * *(matrix2 + (x*num_cols)+c );
            }
        }
    }
    
}
