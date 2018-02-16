from unittest import TestCase

from NiaPy.benchmarks.utility import Utility


class TestBenchmarkFunctions(TestCase):

    def setUp(self):
        self.D = 5
        self.array = [0, 0, 0, 0, 0]
        self.array2 = [1, 1, 1, 1, 1]
        self.array3 = [420.9687, 420.9687, 420.9687, 420.9687, 420.9687]

    def test_rastrigin(self):
        rastrigin = Utility().get_benchmark('rastrigin')
        fun = rastrigin.function()
        self.assertEqual(fun(self.D, self.array), 0.0)

    def test_rosenbrock(self):
        rosenbrock = Utility().get_benchmark('rosenbrock')
        fun = rosenbrock.function()
        self.assertEqual(fun(self.D, self.array2), 0.0)

    def test_griewank(self):
        griewank = Utility().get_benchmark('griewank')
        fun = griewank.function()
        self.assertEqual(fun(self.D, self.array), 0.0)

    def test_sphere(self):
        sphere = Utility().get_benchmark('sphere')
        fun = sphere.function()
        self.assertEqual(fun(self.D, self.array), 0.0)

    def test_ackley(self):
        ackley = Utility().get_benchmark('ackley')
        fun = ackley.function()
        self.assertEqual(round(fun(self.D, self.array)), 0.0)

    def test_schwefel(self):
        schwefel = Utility().get_benchmark('schwefel')
        fun = schwefel.function()
        self.assertAlmostEqual(round(fun(self.D, self.array3)), 0.0)

    def test_schwefel221(self):
        schwefel221 = Utility().get_benchmark('schwefel221')
        fun = schwefel221.function()
        self.assertEqual(fun(self.D, self.array), 0.0)

    def test_schwefel222(self):
        schwefel222 = Utility().get_benchmark('schwefel222')
        fun = schwefel222.function()
        self.assertEqual(fun(self.D, self.array), 0.0)

    def test_whitley(self):
        whitley = Utility().get_benchmark('whitley')
        fun = whitley.function()
        self.assertEqual(fun(self.D, self.array2), 0.0)
