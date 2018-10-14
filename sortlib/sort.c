#include <stdlib.h>
#include <stdio.h>

int compare_int(const void *const a, const void *const b) {
	return (*(int *)a) - (*(int *)b);
}

int compare_float(const void *const a, const void *const b) {
	return (*(float *)a) - (*(float *)b);
}

inline static void __swap(
	register void *a, 
	register void *b, 
	register size_t size_memb) {

	register unsigned char c;
	while(size_memb--) {
		c = *(unsigned char *)a;
		*(unsigned char *)(a++) = *(unsigned char *)b;
		*(unsigned char *)(b++) = c;
	}
}

void _quicksort(
	void *const vector, 
	const long start_index,
	const long final_index, 
	const size_t size_memb,
	int (*const compare_func)(const void *, const void *)) {

	register size_t tail_aux_ind = start_index, 
		head_aux_ind = final_index;

	if (start_index >= final_index)
		return;

	//register const void *const pivot = vector + size_memb * ((final_index + start_index)/(2 * size_memb));
	register const void *const pivot = vector + start_index;

	while (tail_aux_ind <= head_aux_ind) {
		while(tail_aux_ind < final_index &&
			compare_func(vector + tail_aux_ind, pivot) < 0) {
			tail_aux_ind += size_memb;
		}

		while(head_aux_ind > start_index &&
			compare_func(vector + head_aux_ind, pivot) > 0) {
			head_aux_ind -= size_memb;
		}

		if (tail_aux_ind <= head_aux_ind) {
			__swap(vector + head_aux_ind, vector + tail_aux_ind, size_memb);
			tail_aux_ind += size_memb;
			head_aux_ind -= size_memb;
		}
	}

	_quicksort(vector, start_index, head_aux_ind, size_memb, compare_func);
	_quicksort(vector, tail_aux_ind, final_index, size_memb, compare_func);
}

void quicksort(
	void *const vector, 
	const size_t num_memb, 
	const size_t size_memb,
	int (*const compare_func)(const void *, const void *)) {

	_quicksort(vector, 0, size_memb*(num_memb-1), size_memb, compare_func);
}


typedef int test_type;
#define printf_mask "%d "
#define vec_size 50000000
#define min_val (-1000)
#define max_val (+1000)
int main(int argc, char *argv[]) {
	test_type *array = malloc(sizeof(test_type) * vec_size);

	srand(123);
	for (register size_t i = 0; i < vec_size; i++) {
		array[i] = (test_type) (1.0 * (max_val - min_val) * rand()/(1.0 * RAND_MAX)) + min_val;
	}
	printf("\n");

	quicksort(array, vec_size, sizeof(test_type), compare_int);
	printf("\nfinished.\n");
	

	/*	
	for (register size_t i = 0; i < vec_size; ++i) {
		printf(printf_mask, array[i]);
	}
	*/
	
	puts("\n");
	free(array);
	return 0;
}
