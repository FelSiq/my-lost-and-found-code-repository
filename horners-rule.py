def horner(x, pol_coeffs):
	"""Evaluates a polynomial in O(n) time complexity."""
	res = 0.0
	for i in range(len(pol_coeffs)):
		res = pol_coeffs[-i-1] + x * res
	return res

if __name__ == "__main__":
	ans = horner(5.7, [1, 2, 4, 0, 0, -3, 2])
	print("result =", ans)
	ans = horner(3, [0, 0, 1, 0, 0, 10, 20])
	print("result =", ans)
	ans = horner(1, [0, 0, 1, 0, 0, 10, 20])
	print("result =", ans)
	
