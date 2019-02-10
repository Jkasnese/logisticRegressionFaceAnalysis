/*
 * Device code for linear logistic.
 * Algorithm extracted from:
 * https://mmlind.github.io/Using_Logistic_Regression_to_solve_MNIST/
 * @file opencl.c
 * @author Manuella Vieira e Guilherme Lopes
 * @date 09/02/2019
 * 
 */

#define num_pixels = 128*128 + 1 // IMG_WIDTH * HEIGHT + BIAS
#define update = 0.01 / 4487 // LEARNING RATE / NUMBER_OF_IMAGES

__kernel void train(       
   __global float* training,      
   __global float* test,          
   __global float* labels_train,  
   __global float* labels_test,   
   __global float* weights,
   __global float* loss,        
   __global float test_accuracy, 
   __global float precision, 
   __global float recall, 
   __global float fone, 
   const int num_test_samples,  
   const int num_of_epochs)      

{ 

    // Get thread IDs
    int thread_id = get_global_id(0); 
    int img = thread_id * num_pixels;

    // Auxiliary variables
    float temp;
    float aux;

    __local float gradient[num_pixels]; 

    for (int epochs=0; epochs<num_of_epochs; epochs++){

        // Zeroing gradients from previous epoch
        for (int i = thread_id; i < num_pixels; i += get_global_size(0)) {
            gradient[i] = 0;
        }

        temp = 0; 
        for (int x=0; x<num_pixels; x++){ 
            temp += training[img+x] * weights[x]; 
        } 
        /** - Calculates logistic hypothesis */ 
        temp = 1 / (1 + (exp( -1.0 * temp)) ); 
 
        /** - Computes loss function */ 
        aux = labels_train[r]*log(temp) + (1 - labels_train[r])*log(1-temp); 
        atomic_sub(loss[epochs], aux);
 
        /** - Computes the difference between label and hypothesis */ 
        aux = labels_train[r] - temp; 
        
        // Make sure all work-items/threads have finished their calculation before updating gradient
        // to prevent a thread from updating their gradient and then some other thread zeroying it's gradient.
        barrier(CLK_LOCAL_MEM_FENCE);

        /** - Computes current gradient */ 
        for (int x=0; x<num_pixels; x++){ 
            atomic_add(gradient[x], training[img + x] * aux);
        }

        // Make sure all work-items/threads have finished calculating their gradient, before updating weights
        barrier(CLK_LOCAL_MEM_FENCE);

        /** - Updates weights */ 
        for (int i= thread_id; i<num_pixels; i += get_global_size(0)){ 
            weights[i] += update * gradient[i]; 
        } 
    } 
 
    // CALCULATE TEST METRICS 
     // Zeroing variables to hold metrics stats: 
    __local int fp = 0, tp = 0, tn = 0, fn = 0; 
     
    /** - Generate hypothesis values for the test set */ 
    if (thread_id =< num_test_samples) {
        temp = 0; 
        for (int x=0; x<num_pixels; x++){ 
            temp += test[img+x] * weights[x]; 
        }
         
        // Calculate logistic hypothesis 
        temp = 1 / (1 + (exp( -1.0 * temp)) ); 
     
        // Compute the difference between label and hypothesis & 
        //  accuracy on training set & 
        //  loss function & 
        //  metrics (accuracy, precision, recall, f1) 
        
        if (labels_test[thread_id] == 1.0){ 
            if (temp < 0.5){ 
                // FP 
                atomic_add(fp, 1); 
            } else { 
                // TP 
                atomic_add(tp, 1); 
            } 
        } else { 
            if (temp < 0.5){ 
                // TN 
                atomic_add(tn, 1); 
            } else { 
                 // FN 
                atomic_add(fn, 1); 
            }
        }
         
        test_accuracy = ((float) (tp + tn))/ num_test_samples; 
        precision = ((float) tp) / (tp+fp); 
        recall = ((float) tp) / (tp + fn); 
        fone = 2*((precision*recall) / (precision + recall)); 
    }
} 