/*
 * Device code for linear logistic.
 * Algorithm extracted from:
 * https://mmlind.github.io/Using_Logistic_Regression_to_solve_MNIST/
 * @file opencl.c
 * @author Manuella Vieira e Guilherme Lopes
 * @date 09/02/2019
 * 
 * Added num of samples to arguments
 * Change number of work-items (in multiples of 32) to check for performance
 */

#define num_pixels 16385 // IMG_WIDTH * HEIGHT + BIAS
#define update 0.01 / 4487 // LEARNING RATE / NUMBER_OF_IMAGES





__kernel void train(       
   __constant float* training,      
   __constant float* test,          
   __constant float* labels_train,  
   __constant float* labels_test,   
   volatile __global float* weights,
   volatile __global float* gradient,
   volatile __global float* loss,
   __local float* local_loss,
   __local float* dot_product,        
   __global float* test_accuracy, 
   __global float* precision, 
   __global float* recall, 
   __global float* fone, 
   const int num_test_samples,
   const int num_of_samples,  
   const int num_of_epochs)     
{ 

    // Get thread IDs
    int id_x = get_local_id(0); 
    int id_y = get_local_id(1); 
    int img;

    // Auxiliary variables
    float temp;
    float aux;
    float hypothesis;

    //__local float local_loss[get_local_size(1)];
    //__local float dot_product[get_local_size(0)][get_local_size(1)];
    // So local memory used will be y_sz*(1 + x_sz). Must be less than 12284 and max_work_group_size == 1024
    // So possible values for sizes are 32x32, 64x16, 128x8, 256x4, 512x2, 1024x1

    for (int epochs=0; epochs<num_of_epochs; epochs++){

        // Zeroing gradients and loss from previous epoch
        for (uint i = id_x; i < num_pixels; i += get_local_size(0)) {
            gradient[i] = 0;
        }

        local_loss[id_y] = 0;

        // id_y iterates over images
        for (uint r = id_y; r < num_of_samples; r += get_local_size(1)) {
            hypothesis = 0;
            img = r*num_pixels;
            temp = 0; 

            //id_x iterates over pixels
            for (uint x=id_x; x<num_pixels; x+=get_local_size(0)){ 
                // Creating a __local buffer to weights may improve performance
                temp += training[img+x] * weights[x]; 
            } 

            // Each work-item computes part of the image hypothesis, stored in the __local hypothesis array
            dot_product[id_x + id_y*get_local_size(0)] = temp;

            // Barrier to make sure every work-item has already calculated it's part of the hypothesis
            barrier(CLK_LOCAL_MEM_FENCE);

            // Reduce the dot product in order to calculate hypothesis
            // Using parallel reduction to speed up
            for (uint stride = get_local_size(0)/2; stride > 0; stride /= 2) {
                // Barrier to make sure all work-items have written to local memory
                barrier(CLK_LOCAL_MEM_FENCE);

                if (id_x < stride) {
                    dot_product[id_x + id_y*get_local_size(0)] += dot_product[id_x + stride];
                }
            }

            // First element of each "row" (i.e., id_y) contains hypothesis reduction
            hypothesis = dot_product[id_y*get_local_size(0)];

            /** - Calculates logistic hypothesis */ 
            temp = 1 / (1 + (exp( -1.0 * hypothesis)) ); 
     
            /** - Computes loss function */ 
            aux = labels_train[r]*log(temp) + (1 - labels_train[r])*log(1-temp); 

            /** - Computes the difference between label and hypothesis */ 
            aux = labels_train[r] - temp;

            if (0 == id_x) {
                // Each id_y is a picture, so each first thread of each id_y can update the loss of current image
                local_loss[id_y] -= aux;
            }

            /** - Computes current gradient */ 
            for (uint x=id_x; x<num_pixels; x+=get_local_size(0)){ 
                gradient[x] += training[img + x] * aux;
            }
        }

        // Make sure all work-items/threads have finished calculating their gradient, before updating weights
        barrier(CLK_LOCAL_MEM_FENCE);

        /** - Updates weights */
        if ( 0 == id_y){ 
            for (uint i= id_x; i<num_pixels; i+= get_local_size(0)){ 
                weights[i] += update * gradient[i]; 
            } 
        }

        // Update loss epoch by reducing local_loss array
        if(0 == id_y){
            for (uint stride = get_local_size(1)/2; stride > 0; stride /= 2) {
                // Barrier to make sure all work-items have written to local memory
                barrier(CLK_LOCAL_MEM_FENCE);

                if (id_x < stride) {
                    local_loss[id_x] -= local_loss[id_x+stride];
                }
            }

            if (0 == id_x){
                loss[epochs] = local_loss[0];
            }
        }
    } 
 
    // CALCULATE TEST METRICS 
     // Zeroing variables to hold metrics stats: 
    __local int fp, tp, tn, fn; 

    fp = 0; 
    tp = 0; 
    tn = 0; 
    fn = 0;

    // Make sure all weights have been updated by all the work-items
    barrier(CLK_LOCAL_MEM_FENCE);
     
    /** - Generate hypothesis values for the test set */ 
    for (int r = id_y; r<num_test_samples; r+=get_local_size(1)) {
        temp = 0;
        img = r*num_pixels;
        for (int x=id_x; x<num_pixels; x+=get_local_size(0)){ 
            temp += test[img+x] * weights[x]; 
        }

        // Each work-item computes part of the image hypothesis, stored in the __local hypothesis array
        dot_product[id_y*get_local_size(0) + id_x] = temp;

        barrier(CLK_LOCAL_MEM_FENCE);

        // Reduce the dot product in order to calculate hypothesis
        // Using parallel reduction to speed up
        for (uint stride = get_local_size(0)/2; stride > 0; stride /= 2) {
            // Barrier to make sure all work-items have written to local memory
            barrier(CLK_LOCAL_MEM_FENCE);

            if (id_x < stride) {
                dot_product[id_x + id_y*get_local_size(0)] += dot_product[id_x + stride];
            }
        }

        // First element of each "row" (i.e., id_y) contains hypothesis reduction
        hypothesis = dot_product[id_y*get_local_size(0)];
         
        // Calculate logistic hypothesis 
        temp = 1 / (1 + (exp( -1.0 * hypothesis)) ); 
     
        // Compute the difference between label and hypothesis & 
        //  accuracy on training set & 
        //  loss function & 
        //  metrics (accuracy, precision, recall, f1) 
        // Since multiple work-items are used to calculate a single img prediction,
        // we use only one of such work-items to update the test metrics
        if (id_x == 0){
            if (labels_test[r] == 1.0){ 
                if (temp < 0.5){ 
                    // FP 
                    atomic_add(&fp, 1); 
                } else { 
                    // TP 
                    atomic_add(&tp, 1); 
                } 
            } else { 
                if (temp < 0.5){ 
                    // TN 
                    atomic_add(&tn, 1); 
                } else { 
                     // FN 
                    atomic_add(&fn, 1); 
                }
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    float ta, prec, rec, fo;

    ta = ((float) (tp + tn))/ num_test_samples; 
    *test_accuracy = ta;
    prec = ((float) tp) / (tp+fp); 
    *precision = prec;
    rec = ((float) tp) / (tp + fn); 
    *recall = rec;
    fo = 2*((prec*rec) / (prec + rec)); 
    *fone = fo; 
} 