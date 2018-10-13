"""
	Few facts about Minkowski distance:

	Given the order of the distance p:

	1-Minkowski distance (p = 1) = Manhattan distance
	2-Minkowski distance (p = 2) = Euclidean distance
	+inf-Minkowski distance (p -> +inf) = max(inst_a - inst_b)
	-inf-Minkowski distance (p -> -inf) = min(inst_a - inst_b)
"""

def minkowskidist(inst_a, inst_b, p=2.0):
	return (sum((abs(inst_a - inst_b))**p))**(1.0/p)

if __name__ == "__main__":
	from numpy import array
	import sys

	if len(sys.argv) < 3:
		print("usage:", sys.argv[0], 
			"<a_coords> <b_coords>",
			"\n\t[-p distance_order_factor,",
				"default is 2.0 like Euclidean Distance]",
			"\n\t[-sep, default is \",\"]")
		exit(1)

	try:
		sep = sys.argv[1 + sys.argv.index("-sep")]
	except:
		sep = ","

	try:
		p = float(sys.argv[1 + sys.argv.index("-p")])
	except:
		p = 2.0
	
	inst_a = array(list(map(float, sys.argv[1].split(sep=sep))))
	inst_b = array(list(map(float, sys.argv[2].split(sep=sep))))

	dist = minkowskidist(inst_a, inst_b, p=p)

	print("Distance:", dist)
