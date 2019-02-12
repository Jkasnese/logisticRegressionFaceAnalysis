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


inline void atomicAdd_g_f(volatile __global float *addr, float val)
   {
       union {
           unsigned int u32;
           float        f32;
       } next, expected, current;
    current.f32    = *addr;
       do {
       expected.f32 = current.f32;
           next.f32     = expected.f32 + val;
        current.u32  = atomic_cmpxchg( (volatile __global unsigned int *)addr, 
                               expected.u32, next.u32);
       } while( current.u32 != expected.u32 );
   }


__kernel void train(       
   __constant float* training,      
   __constant float* test,          
   __constant float* labels_train,  
   __constant float* labels_test,   
   volatile __global float* weights,
   volatile __global float* gradient,
   volatile __global float* loss,    
   __local float* local_loss,     
   __global float* test_accuracy, 
   __global float* precision, 
   __global float* recall, 
   __global float* fone, 
   const int num_test_samples,
   const int num_of_samples,  
   const int num_of_epochs)     
{ 

    // Get thread IDs
    int thread_id = get_global_id(0); 
    int img;

    // Auxiliary variables
    float temp;
    float aux;

    if(thread_id < num_of_samples){

      for (int epochs=0; epochs<num_of_epochs; epochs++){

          // Zeroing gradients and losses from previous epoch
          for (int i = thread_id; i < num_pixels; i += get_global_size(0)) {
              gradient[i] = 0;
          }

          local_loss[thread_id] = 0;


          img = thread_id*num_pixels;
          temp = 0; 
          for (int x=0; x<num_pixels; x++){ 
              temp += training[img+x] * weights[x]; 
          } 
          /** - Calculates logistic hypothesis */
          temp = 1 / (1 + (exp( -1.0 * temp)) ); 
   
          /** - Computes loss function */ 
          aux = labels_train[thread_id]*log(temp) + (1 - labels_train[thread_id])*log(1-temp); 
          local_loss[thread_id] -= aux;  
          /** - Computes the difference between label and hypothesis */ 
          aux = labels_train[thread_id] - temp; 
          
          // Make sure all work-items/threads have finished their calculation before updating gradient
          // to prevent a thread from updating their gradient and then some other thread zeroying it's gradient.
          barrier(CLK_LOCAL_MEM_FENCE);

          /** - Computes current gradient */ 
          for (int x=0; x<num_pixels; x++){ 
              // This operation should be an atomic_add. However, NVIDIA doesn't currently supports it.
              // To efficiency measures, we let it be a normal add, since atomic_add and + takes about the same time
              gradient[x]+= (training[img+x] * aux);
              //barrier(CLK_LOCAL_MEM_FENCE);
              //atomicAdd_g_f(&gradient[x], training[img + x] * aux);
          }

          // Make sure all work-items/threads have finished calculating their gradient, before updating weights
          barrier(CLK_LOCAL_MEM_FENCE);

          /** - Updates weights */ 
          for (int i= thread_id; i<num_pixels; i += get_global_size(0)){ 
              weights[i] += update * gradient[i];
          } 

         // Update loss epoch by reducing local_loss array
          if (thread_id == 0){
            float epoch_loss = 0;
            for (int i = 0; i < get_global_size(0); i++) {
              epoch_loss -= local_loss[i];
            }
            loss[epochs] = epoch_loss;
          }

          barrier(CLK_LOCAL_MEM_FENCE);

      } 
   
      // CALCULATE TEST METRICS 
       // Zeroing variables to hold metrics stats: 
      __local int fp, tp, tn, fn; 
       

       if(thread_id < num_test_samples){

      /** - Generate hypothesis values for the test set */ 
          temp = 0;
          img = thread_id*num_pixels; 
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
  }
} 