import typing
import unittest
import numpy as np

from csp import CSP


class TestCSP(unittest.TestCase):

    def test_search_horizontal_grid(self):
        """
        Test if search works on a 1xn grid.
        """
        horizontal_groups = [[(0, 0), (0, 1)], [(0, 1), (0, 2)], [(0, 2), (0, 3)], [(0, 3), (0, 4)]]
        groups = horizontal_groups
        # every constraint is of the form (sum, count). so every group must sum to 3 and every number may only occur once per group
        constraints = [(3, 1), (3, 1), (3, 1), (3, 1)]

        # starting with 1
        valid_grid = np.array([[1, 0, 0, 0, 0]])
        csp = CSP(valid_grid, numbers=set([1, 2]), groups=groups, constraints=constraints)
        result = csp.start_search()

        solution_grid = np.array([[1, 2, 1, 2, 1]])

        self.assertTrue(np.all(solution_grid == result))

        # starting with 2
        valid_grid = np.array([[2, 0, 0, 0, 0]])
        csp = CSP(valid_grid, numbers=set([1, 2]), groups=groups, constraints=constraints)
        result = csp.start_search()

        solution_grid = np.array([[2, 1, 2, 1, 2]])

        self.assertTrue(np.all(solution_grid == result))

    def test_search_vertical_grid(self):
        """
        Test if search works on a mx1 grid.
        """
        vertical_groups = [[(0, 0), (1, 0)], [(1, 0), (2, 0)], [(2, 0), (3, 0)], [(3, 0), (4, 0)]]
        groups = vertical_groups
        # every constraint is of the form (sum, count). so every group must sum to 3 and every number may only occur once per group
        constraints = [(3, 1), (3, 1), (3, 1), (3, 1)]

        # starting with 1
        valid_grid = np.array([[1], [0], [0], [0], [0]])
        csp = CSP(valid_grid, numbers=set([1, 2]), groups=groups, constraints=constraints)
        result = csp.start_search()

        solution_grid = np.array([[1], [2], [1], [2], [1]])

        self.assertTrue(np.all(solution_grid == result))

        # starting with 2
        valid_grid = np.array([[2], [0], [0], [0], [0]])
        csp = CSP(valid_grid, numbers=set([1, 2]), groups=groups, constraints=constraints)
        result = csp.start_search()

        solution_grid = np.array([[2], [1], [2], [1], [2]])

        self.assertTrue(np.all(solution_grid == result))

    def test_search_empty_grid(self):
        """
        Test if search works on an empty grid with multiple solutions.
        """
        horizontal_groups = [[(0, 0), (0, 1)], [(1, 0), (1, 1)]]
        vertical_groups = [[(0, 0), (1, 0)], [(0, 1), (1, 1)]]
        groups = horizontal_groups + vertical_groups
        # every constraint is of the form (sum, count). so every group must sum to 3 and every number may only occur once per group
        constraints = [(3, 1), (3, 1), (3, 1), (3, 1)]

        empty_grid = np.array([
            [0, 0],
            [0, 0]
        ])
        csp = CSP(empty_grid, numbers=set([1, 2]), groups=groups, constraints=constraints)
        result = csp.start_search()

        solutions = [
            np.array([
                [1, 2],
                [2, 1]
            ]),
            np.array([
                [2, 1],
                [1, 2]
            ])
        ]

        self.assertTrue(any(np.all(solution_grid == result) for solution_grid in solutions))
