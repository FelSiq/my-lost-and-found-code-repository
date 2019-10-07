"""Monte Carlo approximation algorithm for the NP-Complete 3-CNF Satisfiability problem."""
import typing as t

import numpy as np


def randomized_3_cnf(
        cnf_formula: t.Sequence[t.Tuple[int, int, int]],
        repetitions: int = 10,
        return_solutions: bool = False,
        num_var: t.Optional[int] = None,
        random_state: t.Optional[int] = None
) -> t.Union[float, t.Tuple[float, t.Set[t.Tuple[bool, ...]]]]:
    """Monte Carlo validation for how many clauses in a 3-CNF formula are satisfiable.

    The 3-CNF Satisfiability is a NP-Complete problem, which means that a
    deterministic polynomial exact solution to it is unlikely to exists.

    This implementation uses Monte Carlo validation, assigning True or False
    (each with probability 0.5 from a uniform distribution) to every variable
    in the given ``cnf_formula``, and then computes an approximation to the
    expected value to how many clauses are satisfiable in the given formula.

    Arguments
    ---------
    cnf_formula : :obj:`Sequence` of :obj:`Tuples` of :obj:`int`
        A 3-CNF formula, which must be any sequence of tuples. Each tuple
        represents a distict clause and, therefore, must have three numbers,
        each one representing a distinct variable. Use negative values to
        imply that the negation of that variable must be used.

        Example:
        >>> formula = [(1, -3, 2), (2, 4, -1)]

        Represents the following 3-CNF formula:

        $(x_{1} OR NOT x_{3} OR x_{2}) AND (x_{2} OR x_{4} OR NOT x_{1})$

        Which has four different variables and two clauses.

    repetitions : :obj:`int`, optional
        Number of repetitions for the Monte Carlo procedure.

    return_solutions : :obj:`bool`, optional
        If True, return unique assignments randomly generated that satisfy
        ``cnf_formula``.

    num_var : :obj:`int`, optional
        If given, this procedure will skip the steps to verify how many
        variables are in the given ``given_formula``, which helps this
        function runs faster.

    random_state : :obj:`int`, optional
        If given, set the numpy random seed before the first random
        assignment.

    Notes
    -----
    A ``clause`` of a 3-CNF formula is a pack of three distinct variables (or
    its respective negation) in a OR chain (e.g. (x_{1} OR x_{3} OR NOT X_{2}).
    There must be EXACTLY three distinct variables per clause.

    Every clause in a CNF (which stands for ``Conjuntive Normal Form``) is
    separated by an AND, which means that every clause must be satisfied (if
    possible) in order to the formula as a whole be satisfied.

    Note that there exists 3-CNF formulas that are not satisfiable (i.e. does
    not exists a input which makes the entire formula outputs True). The
    3-CNF Problem is a decision problem which poses the question: ``given an
    instance (some boolean formula in the 3-CNF format), is there any combination
    of boolean input values which, when assigned to each one of the variables
    in that formula, it turns out to be True or, more technically, be satisfied?``

    As said before, this problem is NP-Complete (it is in the intersection of
    the complexity classes NP and NP-Hard), so thats why a approximate solution
    is useful.
    """
    if num_var is None:
        num_var = 0
        for clause in cnf_formula:
            num_var = max(num_var, *clause)

    if num_var <= 0:
        raise ValueError("Number of variables ('num_var') must be positive.")

    if repetitions <= 0:
        raise ValueError(
            "Number of repetitions ('reperitions') must be positive. Got {}.".
            format(repetitions))

    if random_state is not None:
        np.random.seed(random_state)

    num_assign_satisfied = 0

    num_clauses = len(cnf_formula)

    shifted_abs_formula = np.abs(cnf_formula) - 1
    sign_formula = np.sign(cnf_formula)

    assignments = np.random.choice([-1, 1], size=(repetitions, num_var))

    if return_solutions:
        assignments_sat = set()

    for assignment in assignments:
        cur_clauses_satisfied = 0

        for mod_clause, sign_clause in zip(shifted_abs_formula, sign_formula):
            cur_clauses_satisfied += np.any(
                assignment[mod_clause] == sign_clause)

        num_assign_satisfied += cur_clauses_satisfied

        if return_solutions and cur_clauses_satisfied == num_clauses:
            assignments_sat.update([tuple(assignment == 1)])

    avg_sat = num_assign_satisfied / repetitions

    if return_solutions:
        return avg_sat, assignments_sat

    return avg_sat


def _test_01():
    """Test 01 the approximation algorithm with some 3-CNF formula."""
    formula = [
        (1, 2, -3),
        (3, 4, -1),
        (-1, -2, 5),
        (2, 4, 3),
        (3, -2, -4),
    ]

    num_rep = 128
    random_state = 16

    avg_satis, solutions = randomized_3_cnf(
        formula,
        repetitions=num_rep,
        random_state=random_state,
        return_solutions=True)

    print("Average satisfiability: {} (performed {} reps.)".format(
        avg_satis, num_rep))

    print("Theoretical expected satisfiability (see Cormen et al. for proof):",
          7 / 8 * len(formula))

    if solutions:
        print("Found {} solutions:".format(len(solutions)))
        for sol in solutions:
            print(sol)


def _test_02():
    """Test 02 the approximation algorithm with some 3-CNF formula."""
    num_rep = 4096
    random_state = 16
    num_var = 20
    num_clauses = 50

    np.random.seed(random_state)
    formula = np.random.randint(-num_var, num_var + 1, size=(num_clauses, 3))
    zeros = formula == 0
    formula[zeros] = np.random.randint(
        1, num_var + 1, size=np.sum(zeros)) * np.random.choice(
            [1, -1], size=np.sum(zeros))

    avg_satis, solutions = randomized_3_cnf(
        formula,
        repetitions=num_rep,
        random_state=random_state,
        return_solutions=True)

    print("Average satisfiability: {} (performed {} reps.)".format(
        avg_satis, num_rep))

    print("Theoretical expected satisfiability (see Cormen et al. for proof):",
          7 / 8 * len(formula))

    if solutions:
        print("Found {} solutions:".format(len(solutions)))
        for sol in solutions:
            print(sol)


if __name__ == "__main__":
    _test_01()
    _test_02()
