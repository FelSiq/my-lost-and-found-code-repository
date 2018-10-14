#include <stdlib.h>
#include <stdio.h>

int compare_int(const void *const a, const void *const b) {
	return (*(int *)a) - (*(int *)b);
}

int compare_long(const void *const a, const void *const b) {
	return (*(long *)a) - (*(long *)b);
}

int compare_byte(const void *const a, const void *const b) {
	return (*(unsigned char *)a) - (*(unsigned char *)b);
}

int compare_float(const void *const a, const void *const b) {
	return (*(float *)a) - (*(float *)b);
}

int compare_double(const void *const a, const void *const b) {
	return (*(double *)a) - (*(double *)b);
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
	int (* const compare_func)(const void *, const void *)) {

	register size_t tail_aux_ind = start_index, 
		head_aux_ind = final_index;

	if (start_index >= final_index)
		return;

	register const void *const pivot = vector + 
		size_memb * ((final_index + start_index)/(2 * size_memb));

	while (tail_aux_ind <= head_aux_ind) {
		while(compare_func(vector + tail_aux_ind, pivot) < 0)
			tail_aux_ind += size_memb;

		while(compare_func(vector + head_aux_ind, pivot) > 0)
			head_aux_ind -= size_memb;

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

static inline void __copy(void *const dest, void *const src, const size_t size) {
	register size_t i = 0;
	while (i < size) {
		*(unsigned char *)(dest + i) = *(unsigned char *)(src + i);
		i++;
	}
}

void _mergesort(
	void *const vector,
	const long start_index,
	const long final_index,
	const size_t size_memb,
	int (* const compare_func)(const void *, const void *),
	unsigned char *buffer) {

	if (start_index >= final_index)
		return;

	// Split phase
	register long middle = size_memb * ((final_index + start_index)/(2 * size_memb));

	_mergesort(vector, start_index, middle, size_memb, compare_func, buffer);
	_mergesort(vector, middle+size_memb, final_index, size_memb, compare_func, buffer);

	// Merge Phase
	register long i = start_index, j = middle+size_memb, counter = 0;

	while (i <= middle || j <= final_index) {
		if (i > middle || (j <= final_index && 
			compare_func(vector + i, vector + j) > 0)) {

			__copy(buffer + counter, vector + j, size_memb);
			counter += size_memb;
			j += size_memb;

		} else {
			__copy(buffer + counter, vector + i, size_memb);
			counter += size_memb;
			i += size_memb;
		}
	}

	for (i = start_index; i <= final_index; i += size_memb)
		__copy(vector + i, buffer + i - start_index, size_memb);

}

void mergesort(
	void *const vector,
	const size_t num_memb, 
	const size_t size_memb,
	int (*const compare_func)(const void *, const void *)) {

	unsigned char *buffer = malloc(sizeof(unsigned char) *
		(num_memb * size_memb));
	_mergesort(vector, 0, size_memb*(num_memb-1), size_memb, compare_func, buffer);
	free(buffer);
}

#define HEAP_NODE_MASTER(NODE) (((NODE) - 1)/2)
#define HEAP_NODE_SONL(NODE) (2*(NODE) + 1)
#define HEAP_NODE_SONR(NODE) (2*(NODE) + 2)

static inline void _heapfy(
	void *const vector,
	const size_t num_memb, 
	const size_t size_memb,
	int (*const compare_func)(const void *, const void *)) {

	register long cur_pos_ind, master_pos_ind;
	register void *cur_pos_key, *master_pos_key;
	for (register long heap_end_ind = 1; 
		heap_end_ind < num_memb; 
		heap_end_ind++) {
		
		cur_pos_ind = heap_end_ind;
		master_pos_ind = HEAP_NODE_MASTER(heap_end_ind);

		cur_pos_key = vector + size_memb * cur_pos_ind;
		master_pos_key = vector + size_memb * master_pos_ind;

		while (master_pos_ind >= 0 &&
			compare_func(cur_pos_key, master_pos_key) > 0) {

			__swap(cur_pos_key, master_pos_key, size_memb);
			cur_pos_ind = master_pos_ind;
			master_pos_ind = HEAP_NODE_MASTER(cur_pos_ind);

			cur_pos_key = vector + size_memb * cur_pos_ind;
			master_pos_key = vector + size_memb * master_pos_ind;
		}
	}
}

static inline void _heapsort_pop(
	void *const vector, 
	size_t heap_size, 
	const size_t size_memb,
	int (*const compare_func)(const void *, const void *)) {

	// Swap first position with last position
	__swap(vector + size_memb * (--heap_size), vector, size_memb);

	// Push new first position through a correct
	// position in heap vector
	register long cur_pos_ind = 0, 
		sonl_index = HEAP_NODE_SONL(cur_pos_ind), 
		sonr_index = HEAP_NODE_SONR(cur_pos_ind), 
		best_son_ind;
	unsigned char *best_son_key, *cur_pos_key;
	
	while (sonl_index < heap_size) {
		best_son_key = vector + size_memb * sonl_index;
		best_son_ind = sonl_index;
		if (sonr_index < heap_size && 
			compare_func(vector + size_memb * sonr_index, best_son_key) > 0) {
			best_son_key = vector + size_memb * sonr_index;
			best_son_ind = sonr_index;
		}
		
		cur_pos_key = vector + size_memb * cur_pos_ind;
		if (compare_func(cur_pos_key, best_son_key) < 0) {
			__swap(cur_pos_key, best_son_key, size_memb);
			cur_pos_ind = best_son_ind;
			sonl_index = HEAP_NODE_SONL(cur_pos_ind);
			sonr_index = HEAP_NODE_SONR(cur_pos_ind);
		} else sonl_index = heap_size;
	}
}

void heapsort(
	void *const vector,
	const size_t num_memb, 
	const size_t size_memb,
	int (*const compare_func)(const void *, const void *)) {

	// Build up a max heap
	_heapfy(vector, num_memb, size_memb, compare_func);

	for (register size_t i = num_memb; i > 1; i--) {
		_heapsort_pop(vector, i, size_memb, compare_func);
	}
}

typedef float test_type;
#define printf_mask "%f "
#define vec_size 50000000
#define min_val (-1000)
#define max_val (+1000)
int main(int argc, char *argv[]) {
	test_type *array = malloc(sizeof(test_type) * vec_size);

	srand(123);
	for (register size_t i = 0; i < vec_size; i++) {
		array[i] = (test_type) (1.0 * (max_val - min_val) * rand()/(1.0 * RAND_MAX)) + min_val;
//		printf(printf_mask, array[i]);
	}
	puts("\n");

	quicksort(array, vec_size, sizeof(test_type), compare_float);
	puts("\nfinished.\n");

/*	
	for (register size_t i = 0; i < vec_size; ++i) {
		printf(printf_mask, array[i]);
	}
*/	
	
	puts("\n");
	free(array);
	return 0;
}
