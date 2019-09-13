import sys

def add(s1, s2):

	l1 = len(s1)
	l2 = len(s2)
	max_size = max(l1, l2)

	s1 = "0" * (1 + max(0, max_size - l1)) + s1
	s2 = "0" * (1 + max(0, max_size - l2)) + s2

	res = ""
	carry = "0"
	for i in range(max_size, -1, -1):
		b1 = s1[i] == "1"
		b2 = s2[i] == "1"
		aux = b1 ^ b2 ^ (carry == "1")
		carry = "1" if ((b1 and b2) or \
			(b1 and carry == "1") or \
			(b2 and carry == "1")) else "0"
		res = ("1" if aux else "0") + res

	while res[0] == "0":
		res = res[1:]

	return res
		

if __name__ == "__main__":
	if len(sys.argv) < 3:
		print("usage:", sys.argv[0], "<s1> <s2>")
		exit(1)

	s1 = sys.argv[1]
	s2 = sys.argv[2]

	res = add(s1, s2)

	print("Result:", res, 
		"\nReal proof:", 
		int(s1, 2), "+", int(s2, 2), "=", int(res, 2), 
		"(", int(s1, 2) + int(s2, 2) == int(res, 2), ")")
