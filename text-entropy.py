import sys
import math

def entropyCalculus(freqs, relative=False):
	normalization = 1
	if not relative:
		normalization = sum(freqs)
	return -(sum([n/normalization * math.log(n/normalization, 2) for n in freqs]))

def fileEntropy(filepath):
	with open(filepath, 'r') as fp:
		absFreqs = {}
		for line in fp:
			for char in line:
				if char not in absFreqs:
					absFreqs[char] = 0
				absFreqs[char] += 1
		return entropyCalculus(list(absFreqs.values()), relative=False)

if __name__ == '__main__':
	if len(sys.argv) <= 1:
		print('usage:', sys.argv[0], '<filepath>')
		exit(1)
	fileEntropy = fileEntropy(sys.argv[1])
	print(fileEntropy)
