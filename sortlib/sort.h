#ifndef __SORT_H__
#define __SORT_H__

int compare_int(const void *const a, const void *const b);
int compare_long(const void *const a, const void *const b);
int compare_byte(const void *const a, const void *const b);
int compare_float(const void *const a, const void *const b);
int compare_double(const void *const a, const void *const b);

void quicksort(
	void *const vector, 
	const size_t num_memb, 
	const size_t size_memb,
	int (*const compare_func)(const void *, const void *));

void mergesort(
	void *const vector,
	const size_t num_memb, 
	const size_t size_memb,
	int (*const compare_func)(const void *, const void *));

void heapsort(
	void *const vector,
	const size_t num_memb, 
	const size_t size_memb,
	int (*const compare_func)(const void *, const void *));

#endif
