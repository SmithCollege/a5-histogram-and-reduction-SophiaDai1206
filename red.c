//CPU Reduction
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <sys/time.h>

double get_clock() {
    struct timeval tv;
    int ok = gettimeofday(&tv, NULL);
    if (ok < 0) {
        printf("gettimeofday error\n");
        return -1.0; 
    }
    return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}


int reduce_sum(int *arr, int size) {
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
    return sum;
}

int product(int *arr, int size) {
    int product = 1;
    for (int i = 0; i < size; i++) {
        product *= arr[i];
    }
    return product;
}

int reduce_min(int *arr, int size) {
    int min = INT_MAX;
    for (int i = 0; i < size; i++) {
        if (arr[i] < min) {
            min = arr[i];
        }
    }
    return min;
}

int reduce_max(int *arr, int size) {
    int max = INT_MIN;
    for (int i = 0; i < size; i++) {
        if (arr[i] > max) {
            max = arr[i];
        }
    }
    return max;
}

#define SIZE 1000000

int main() {
    int* input = malloc(sizeof(int) * SIZE);

	 for (int i = 0; i < SIZE; i++) {
        input[i] = 1;  
    }

	//sum
	double t0 = get_clock();
	printf("%d\n ", reduce_sum(input, SIZE));
	double t1 = get_clock();
	printf("Time of sum: %f s\n", (t1-t0));
	printf("\n");
		
	//product
	t0 = get_clock();
	printf("%d\n ", product(input, SIZE));
	t1 = get_clock();
	printf("Time of product: %f s\n", (t1-t0));
	printf("\n");
	
	//min
	t0 = get_clock();
	printf("%d\n ", reduce_min(input, SIZE));
	t1 = get_clock();
	printf("Time of min: %f s\n", (t1-t0));
	printf("\n");

    //max
	t0 = get_clock();
	printf("%d\n ", reduce_max(input, SIZE));
	t1 = get_clock();
	printf("Time of max: %f s\n", (t1-t0));
	printf("\n");
	

}