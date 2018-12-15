#!/bin/bash

# Run configuration
NODES=1
THREADS=1
EPOCHS=1000
NUM_IMAGES=4487 #3365 2243

# Run relative path
RELATIVE_PATH=""

# Run results
TOTAL_TIME_ALLREDUCE=0
TOTAL_TIME_BCAST=0
RUN_TIME=0


mkdir results
mkdir results/1_node
mkdir results/2_node
mkdir results/3_node

touch "results/results.csv"
printf "nodes_threads,1000_epochs,2000_epochs,3000_epochs,4000_epochs,5000_epochs,3365_images,2243_images\n" >> "results/results.csv"

# Changing number of nodes to be executed in
for i in 1 2 3
do
	# Changing number of threads
	for j in 1 2 4
	do
		# Print identifier to csv file
		printf "${i}_${j}," >> "results/results.csv"

		# Changing number of epochs
		for epochs in 1000 2000 3000 4000 5000
		do
			RELATIVE_PATH=${i}_node/${j}_${epochs}_${NUM_IMAGES}
			# Running 3 times to get mean + std_dev
			for run in 1 2 3
			do
				mpirun -np $i -machinefile machines.txt loopmpi.out $epochs $NUM_IMAGES $i $j
				RUN_TIME=$(cat total_time)
				TOTAL_TIME_ALLREDUCE=$(echo "scale=10; ($RUN_TIME+$TOTAL_TIME_ALLREDUCE)" | bc)
			done
			# Calculate mean
			TOTAL_TIME_ALLREDUCE=$(echo "scale=10; ($TOTAL_TIME_ALLREDUCE/3)" | bc)

			# Print to csv file
			printf "${TOTAL_TIME_ALLREDUCE}," >> "results/results.csv"

			## Clean runs
			# Clean shell variables
			TOTAL_TIME_ALLREDUCE=0

			# Clean generated files
			mv "training_acc.txt" "results/${RELATIVE_PATH}_training_acc.txt"
			mv "loss.txt" "results/${RELATIVE_PATH}_loss.txt"
			mv "test_metrics.txt" "results/${RELATIVE_PATH}_test_metrics.txt"
			mv "execution_times" "results/${RELATIVE_PATH}_execution_times"
			mv "run_time" "results/${RELATIVE_PATH}_run_time"
		done
		for num_images in 3365 2243
		do
			RELATIVE_PATH=${i}_node/${j}_${EPOCHS}_${num_images}
			# Running 3 times to get mean + std_dev
			for run in 1 2 3
			do
				mpirun -np $i -machinefile machines.txt loopmpi.out $EPOCHS $num_images $i $j
				RUN_TIME=$(cat total_time)
				TOTAL_TIME_ALLREDUCE=$(echo "scale=10; ($RUN_TIME+$TOTAL_TIME_ALLREDUCE)" | bc)
			done
			# Calculate mean
			TOTAL_TIME_ALLREDUCE=$(echo "scale=10; ($TOTAL_TIME_ALLREDUCE/3)" | bc)

			# Print to csv file
			printf "${TOTAL_TIME_ALLREDUCE}," >> "results/results.csv"

			## Clean runs
			# Clean shell variables
			TOTAL_TIME_ALLREDUCE=0

			# Clean generated files
			mv "training_acc.txt" "results/${RELATIVE_PATH}_training_acc.txt"
			mv "loss.txt" "results/${RELATIVE_PATH}_loss.txt"
			mv "test_metrics.txt" "results/${RELATIVE_PATH}_test_metrics.txt"
			mv "execution_times" "results/${RELATIVE_PATH}_execution_times"
			mv "run_time" "results/${RELATIVE_PATH}_run_time"
		done
		# New line to new iteration on results.csv	
		printf "\n" >> "results/results.csv"
	done
done
