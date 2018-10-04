import sys
import re

def shunting_yard(string, operators_set={
		"+" : 2, 
		"-" : 2, 
		"*" : 3, 
		"^" : 4, 
		"/" : 3}):

	# Preprocessing the input string
	pattern = list(operators_set.keys()) + ["(", ")"]
	if "-" in pattern:
		pattern.remove("-")
		pattern.insert(0, "-")
	pattern = "([" + "\\" + "\\".join(pattern) + "]|[0-9]+)"
	input_array = re.findall(pattern, string)
	
	ans = []
	operator_stack = []

	for c in input_array:
		if c not in operators_set and c != "(" and c != ")":
			ans.append(c)

		elif c in operators_set:
			# Pop operator stack until the top operator has higher
			# priority
			while operator_stack and \
				c in operators_set and \
				operator_stack[-1] in operators_set and \
				operators_set[operator_stack[-1]] > operators_set[c]:
				ans.append(operator_stack.pop())

			operator_stack.append(c)

		elif c == "(":
			operator_stack.append(c)

		elif c == ")":
			# Pop operator stack until founding a matching "("
			while operator_stack[-1] != "(":
				ans.append(operator_stack.pop())

			# Discard ")"
			operator_stack.pop(-1)

	while operator_stack:
		ans.append(operator_stack.pop())

	return ans

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("usage:", sys.argv[0], "<infix notation expression>")
		exit(1)

	ans = shunting_yard(sys.argv[1])

	print(ans)
