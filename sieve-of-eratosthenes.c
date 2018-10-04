#include <stdio.h>

// Enable this to use global array
#define __ENABLE_GLOBAL__ 0

// Habilite this to enable main function of this code
#define __PROGRAM_DRIVER__ 1

#if __ENABLE_GLOBAL__
	#define limit 1000000000
	char marked[limit];
#else
	#include <stdlib.h>
#endif

unsigned long int sieve(
	#if __ENABLE_GLOBAL__ == 0
		register const unsigned long int limit
	#endif
	) {

	register unsigned long int p, j, count = 1;

	if (limit <= 1)
		return 0;
	if (limit == 2)
		return 1;

	#if __ENABLE_GLOBAL__ == 0
		char *restrict marked = malloc(sizeof(char) * (1 + limit));
		for (p = 3; p <= limit; p++)
			marked[p] = 1;
	#endif

	/*
		Multiples of 2 are ignored because the only
		even prime number is 2.
	*/
	
	for (p = 3; p*p <= limit; p += 2) {
		if (
		#if __ENABLE_GLOBAL__ == 0
			marked[p]
		#else
			!marked[p]
		#endif
		) {
			for (j = 2*p; j <= limit; j += p)
			#if __ENABLE_GLOBAL__ == 0
				marked[j] = 0;
			#else
				marked[j] = 1;
			#endif
		}

	}

	for (j = 3; j <= limit; j += 2)
		count +=
		#if __ENABLE_GLOBAL__ == 0
			marked[j];
		#else
			!marked[j];
		#endif

	#if __ENABLE_GLOBAL__ == 0
		free(marked);
	#endif
	
	return count;
}

#if __PROGRAM_DRIVER__
	int main(int argc, char *const argv[]) {
		#if __ENABLE_GLOBAL__ == 0
			if (argc < 2) {
				printf("usage: %s <limit>\n", argv[0]);
				return 1;
			}

			unsigned long int ans = sieve((unsigned long int) atol(argv[1]));
		#else
			unsigned long int ans = sieve();
		#endif

		printf("%lu\n", ans);

		return 0;
	}
#endif
