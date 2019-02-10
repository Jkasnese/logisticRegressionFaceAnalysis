/*
 * Device code for linear logistic.
 * Algorithm extracted from:
 * https://mmlind.github.io/Using_Logistic_Regression_to_solve_MNIST/
 * @file opencl.c
 * @author Manuella Vieira e Guilherme Lopes
 * @date 09/02/2019
 * 
 */

__kernel void train(       
   __global float* training,      
   __global float* test,          
   __global float* labels_train,  
   __global float* labels_test,   
   __global float* weights,        
   __global float test_accuracy,
   __global float precision, 
   __global float recall, 
   __global float fone, 
   const int num_test_samples,  
   const int num_of_epochs)      
{ 
    __local float learning_rate = 0.01;  

    // Get thread IDs
    int local_x = get_local_id(0); 
    int local_y = get_local_id(1); 
    int local_z = get_local_id(2); 

    int group_x = get_group_id(0); 
    int group_y = get_group_id(1); 

    // Pixel identifiers
    int pixel_id = (local_x * 128) + local_y; // 128 is IMG_WIDTH
    int num_pixels = 128 * 128 + 1 // IMG_WIDTH * HEIGHT + BIAS

    // Auxiliary variables
    float temp;

    __local float gradient[num_pixels]; 
    __global float gradient[num_pixels]; 

    for (int epochs=0; epochs<num_of_epochs; epochs++){  
    // Zeroying gradients 
        if (pixel_id < num_pixels) { 
            gradient[pixel_id] = 0; 
        } 
 
        for ( ; r<num_of_training_imgs; r++){ 
            temp = training[pixel_id] * weights[pixel_id]; 

            /** - Calculates logistic hypothesis */ 
            temp = 1 / (1 + (exp( -1.0 * temp)) ); 
     
             /** - Computes accuracy on training set */ 
            if (labels_train[r] == 0.0){ 
                if (temp < 0.5) 
                    right_answers++; 
            } else { 
                if (!(temp < 0.5)) 
                    right_answers++; 
            } 
     
            /** - Computes loss function */ 
            aux = labels_train[r]*log(temp) + (1 - labels_train[r])*log(1-temp); 
            loss += aux; // Acelera se trocar por if/else dos labels? 
     
            /** - Computes the difference between label and hypothesis */ 
            aux = labels_train[r] - temp; 
     
            /** - Computes current gradient */ 
            r_numpixels = r*num_pixels; 
            for (long x=0; x<num_pixels; x++){ 
                gradient[x] += training[r_numpixels + x] * aux; 
            } 
        } 
     
        /** - Updates weights */ 
        for (int i=0; i<num_pixels; i++){ 
            weights[i] += update * gradient[i]; 
        } 
 
    /** - Saves epoch metrics to be plotted later */ 
   } 
 
// CALCULATE TEST METRICS 
 // Zeroing variables to hold metrics stats: 
right_answers = 0; 
int fp = 0, tp = 0, tn = 0, fn = 0; 
 
/** - Generate hypothesis values for the test set */ 
for (int r=0; r<TEST_SAMPLES; r++){ 
    r_numpixels = r*num_pixels; 
    temp = 0; 
    for (long x=0; x<num_pixels; x++){ 
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
 
    test_accuracy = ((float) (tp + tn))/ num_test_samples; 
    precision = ((float) tp) / (tp+fp); 
    recall = ((float) tp) / (tp + fn); 
    fone = 2*((precision*recall) / (precision + recall)); 
} 