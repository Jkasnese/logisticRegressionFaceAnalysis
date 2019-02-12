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

// num_samples is the number of samples each work-group is suposed to calculate
__kernel void gradients(       
   __constant float* training,      
   __constant float* labels_train,  
   volatile __global float* weights,
   volatile __global float* gradient,
   __local float* local_gradient,
   volatile __global float* loss,
   __local float* local_loss,
   __local float* dot_product,
   const int num_samples)        
{ 

    // Get thread IDs
    int id_x = get_local_id(0); 
    int id_y = get_local_id(1); 

    int id_g = get_group_id(0);

    int img;

    // Auxiliary variables
    float temp;
    float aux;
    float hypothesis;

    //__local local_err[get_local_size(1)]; error of samples, to multiply for each pixel and get gradient
    //__local local_weights[get_local_size(0)] --> local_weights works to buffer weights, lowering global accesses
    //__local local_gradient[num_pixels/2]
    //__local float local_loss[get_local_size(1)];
    //__local float dot_product[get_local_size(0)][get_local_size(1)];
    // So local memory used will be y_sz*(1 + x_sz). Must be less than 12284 and max_work_group_size == 1024
    // So possible values for sizes are 32x32, 64x16, 128x8, 256x4, 512x2, 1024x1

    // Zeroing gradients and loss from previous epoch
    for (uint i = id_x; i < num_pixels/2; i += get_local_size(0)) {
        local_gradient[i] = 0;
    }

    local_loss[id_y] = 0;

    // id_y iterates over images
    for (uint r = id_y; r<num_samples; r += get_local_size(1)) {
        hypothesis = 0;
        img = r * id_g * num_pixels;
        temp = 0; 

        local_weights[id_x] = weights[id_x];

        //id_x iterates over pixels
        for (uint x=id_x; x<num_pixels; x+=get_local_size(0)){ 
            // Creating a __local buffer to weights may improve performance
            temp += training[img+x] * local_weights[x]; 
        } 

        // Each work-item computes part of the image hypothesis, stored in the __local hypothesis array
        dot_product[id_x + id_y*get_local_size(0)] = temp;

        // Barrier to make sure every work-item has already calculated it's part of the hypothesis
        barrier(CLK_LOCAL_MEM_FENCE);

        // Reduce the dot product in order to calculate hypothesis
        // Using parallel reduction to speed up
        for (uint stride = get_local_size(0)/2; stride > 0; stride /= 2) {
            if (id_x < stride) {
                dot_product[id_x + id_y*get_local_size(0)] += dot_product[id_x + stride + id_y*get_local_size(0)];
            }

            // Barrier to make sure all work-items have written to local memory
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // First element of each "row" (i.e., id_y) contains hypothesis reduction
        hypothesis = dot_product[id_y*get_local_size(0)];

        /** - Calculates logistic hypothesis */ 
        temp = 1 / (1 + (exp( -1.0 * hypothesis)) ); 
 
        /** - Computes loss function */ 
        aux = labels_train[r]*log(temp) + (1 - labels_train[r])*log(1-temp); 

        if (0 == id_x) {
            // Each id_y is a picture, so each first thread of each id_y can update the loss of current image
            local_loss[id_y] -= aux;
        }

        /** - Computes the difference between label and hypothesis and store on local mem to be used by other work-items*/ 
        aux = labels_train[r] - temp;
        local_err[id_y] = aux;

        /** - Computes first half of gradients */ 
        for (uint x=id_x; x<num_pixels/2; x+=get_local_size(0)){ 
            local_gradient[x] += training[img + x] * local_err[id_y];
        }

        // Make sure all first half of gradients have been calculated and wWrites to global memory
        barrier(CLK_LOCAL_MEM_FENCE);

        if (0 == get_global_id(0)){
            for (uint x=id_x; x<num_pixels/2; x+=get_local_size(0)){ 
                gradient[x] += local_gradient[x]
            }
        }

        // Zero gradients
        for (uint i = id_x; i < num_pixels/2; i += get_local_size(0)) {
            local_gradient[i] = 0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute second half
        for (uint x=id_x + num_pixels/2; x<num_pixels; x+=get_local_size(0)){ 
            local_gradient[x] += training[img + x] * local_err[id_y];
        }

        // Make sure all first half of gradients have been calculated and wWrites to global memory
        barrier(CLK_LOCAL_MEM_FENCE);

        if (0 == get_global_id(0)){
            for (uint x=id_x + num_pixels/2; x<num_pixels; x+=get_local_size(0)){ 
                gradient[x] += local_gradient[x]
            }
        }
        
    }

    // Make sure all work-items/threads have finished calculating their gradient, before updating weights
    barrier(CLK_LOCAL_MEM_FENCE);

    // Update loss epoch by reducing local_loss array
    if(0 == id_y){
        for (uint stride = get_local_size(1)/2; stride > 0; stride /= 2) {
            if (id_x < stride) {
                local_loss[id_x] -= local_loss[id_x+stride];
            }

            // Barrier to make sure all work-items have written to local memory
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (0 == id_x){
            loss[epochs] -= local_loss[0];
        }
    }
} 