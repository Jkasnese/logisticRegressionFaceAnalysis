/*
 * Device code for linear logistic.
 * Algorithm extracted from:
 * https://mmlind.github.io/Using_Logistic_Regression_to_solve_MNIST/
 * @file opencl.c
 * @author Manuella Vieira e Guilherme Lopes
 * @date 09/02/2019
 * 
 * Coalescing reads from global memory.
 * Adding reductions (to global memory) to maintain the algorhitm correctness.
 * Added dot_product in order to calculate images hypothesis, since each work-item calculate portions of hyp
 * TODO: Remove bias from training and testing and see how this improves memory access
 */

#define num_pixels 128*128 + 1 // IMG_WIDTH * HEIGHT + BIAS
#define update 0.01 / 4487 // LEARNING RATE / NUMBER_OF_IMAGES
#define IMG_WIDTH 128


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
    int local_id_x = get_local_id(0); 
    int local_id_y = get_local_id(1); 
    int local_id_z = get_local_id(2); 
    
    // Auxiliary variables
    int img;
    float temp;
    float hypothesis;
    float aux;

    // Because of this __local memory, since max __local is 0xC000 (49152 bytes), the maximum group_size is:
    // size_x*y*z < 12288. So for 32*32 size, we can have 12 images at z. For 64*64, we can have 2 images at z. 
    // __local float dot_product[get_local_size(0)*get_local_size(1)][get_local_size(2)];
    // __local float local_loss[get_local_size(2)]

    for (int epochs=0; epochs<num_of_epochs; epochs++){

        // Zeroing gradients from previous epoch
        for (int i = local_id_y; i < IMG_WIDTH; i += get_local_size(1)) {
            for (int j = local_id_x; j < IMG_WIDTH; j += get_local_size(0)) {
                gradient[IMG_WIDTH * i + j] = 0;
            }
        }

        // Zero bias
            if (local_id_x == get_local_size(0)-1 && local_id_y == get_local_size(1) -1)
                gradient[num_pixels-1];
        
        // Each work-group trains for some images (64 prob, to maximize local gradient access, minimizing global gradient access)
        for (int r=local_id_z; r<num_of_samples; r+=get_local_size(2)) {

            img = r*num_pixels;
            temp = 0;

            // Each work-item calculates dot-product for a pixel, to coalesce global memory access 
            // Maximum expected throughput when local_size(0) == IMG_WIDTH.
            // For bigger local sizes, code has to be rewritten to consider a continum array of pixels, instead of separated by images
            for (int y = local_id_y; y < IMG_WIDTH; y += get_local_size(1)) {
                for (int x= local_id_x; x<IMG_WIDTH; x += get_local_size(0)) {
                     temp += training[img + (y*IMG_WIDTH + x)] * weights[y*IMG_WIDTH + x];
                }
            }

            // Add bias
            if (local_id_x == get_local_size(0)-1 && local_id_y == get_local_size(1) -1)
                temp += weights[num_pixels-1];

            // Each work-item computes part of the image hypothesis, stored in the __local hypothesis array
            //dot_product[local_id_y*get_local_size(0) + local_id_x][local_id_z] = temp;
            dot_product[(local_id_y*get_local_size(0) + local_id_x) * local_id_z] = temp;

            // Barrier to make sure every work-item has already calculated it's part of the hypothesis
            barrier(CLK_LOCAL_MEM_FENCE);

            // Reduce the dot product in order to calculate hypothesis
            for (uint i = 0; i < get_local_size(0)*get_local_size(1); ++i) {
                hypothesis += dot_product[i * local_id_z];
            }

            /** - Calculates logistic hypothesis */ 
            temp = 1 / (1 + (exp( -1.0 * hypothesis)) ); 
     
            /** - Computes loss function */ 
            aux = labels_train[r]*log(temp) + (1 - labels_train[r])*log(1-temp); 
            if (0 == local_id_x && 0 == local_id_y ) {
                // Each id_z is a picture, so each first thread of each id_z can update the loss of current image
                local_loss[local_id_z] -= aux;
            }
     
            /** - Computes the difference between label and hypothesis */ 
            aux = labels_train[r] - temp;
            
            /** - Computes current gradient */
            for (int y=local_id_y; y<IMG_WIDTH; y+=get_local_size(1)) {
                for (int x=local_id_x; x<IMG_WIDTH; x+=get_local_size(0)) {
                    gradient[y*IMG_WIDTH + x] += training[img + (y*IMG_WIDTH + x)] * aux;
                }
            }

            // Add bias
            if (local_id_x == get_local_size(0)-1 && local_id_y == get_local_size(1) -1)
                gradient[num_pixels-1] += aux;
        }

        // Make sure all work-items/threads have finished calculating their gradient, before updating weights
        barrier(CLK_LOCAL_MEM_FENCE);

        /** - Updates weights */ 
        for (int y=local_id_y; y<IMG_WIDTH; y+=get_local_size(1)) {
                for (int x=local_id_x; x<IMG_WIDTH; x+=get_local_size(0)) {
                    weights[y*IMG_WIDTH + x] += update * gradient[img + (y*IMG_WIDTH + x)];
                }
        }

        // Update bias weight
        if (local_id_x == get_local_size(0)-1 && local_id_y == get_local_size(1) -1)
            weights[num_pixels-1] = update * gradient[num_pixels-1];

        // Save epoch loss
        float epoch_loss = 0;
        for (int i = 0; i < get_local_size(2); ++i){
            epoch_loss -= local_loss[i];
        }
        // Update epoch loss 
        // Could be more efficient if grouped in 128 bits, i.e., 4 floats. 4 epochs per turn. 
        loss[epochs] = epoch_loss;
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
    for (int r = local_id_z; r<num_test_samples; r+= get_local_size(2)) {
        temp = 0;
        img = r*num_pixels; 

        for (int y = local_id_y; y < IMG_WIDTH; y += get_local_size(1)) {
            for (int x= local_id_x; x<IMG_WIDTH; x += get_local_size(0)) {
                 temp += test[img + (y*IMG_WIDTH + x)] * weights[y*IMG_WIDTH + x];
            }
        }

        // Add bias
        if (local_id_x == get_local_size(0)-1 && local_id_y == get_local_size(1) -1)
            temp += weights[num_pixels-1];

        // Each work-item computes part of the image hypothesis, stored in the __local hypothesis array
        dot_product[(local_id_y*get_local_size(0) + local_id_x) *local_id_z] = temp;

        // Barrier to make sure every work-item has already calculated it's part of the hypothesis
        barrier(CLK_LOCAL_MEM_FENCE);

        // Reduce the dot product in order to calculate hypothesis
        for (uint i = 0; i < get_local_size(0)*get_local_size(1); ++i) {
            hypothesis += dot_product[i * local_id_z];
        }
         
        // Calculate logistic hypothesis 
        temp = 1 / (1 + (exp( -1.0 * hypothesis)) ); 
     
        // Compute the difference between label and hypothesis & 
        //  accuracy on training set & 
        //  loss function & 
        //  metrics (accuracy, precision, recall, f1) 

        // Since multiple work-items (multiple coordenates x,y per image)
        // are used to calculate a single img prediction,
        // we use only one of such work-items to update the test metrics
        if (0 == local_id_x && 0 == local_id_y ){
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