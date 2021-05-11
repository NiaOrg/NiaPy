# encoding=utf8
from niapy.algorithms.basic import KrillHerdV1, KrillHerdV2, KrillHerdV3, KrillHerdV4, KrillHerdV11
from niapy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark


class KHV1TestCase(AlgorithmTestCase):
    r"""Test case for KrillHerdV1 algorithm.

    Date:
        April 2019

    Author:
        Klemen Berkovič

    See Also:
        * :class:`niapy.tests.test_algorithm.AlgorithmTestCase`
    """

    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = KrillHerdV1

    def test_custom(self):
        kh_custom = self.algo(population_size=10, C_a=2, C_r=0.5, seed=self.seed)
        kh_customc = self.algo(population_size=10, C_a=2, C_r=0.5, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, kh_custom, kh_customc, MyBenchmark())

    def test_griewank(self):
        kh_griewank = self.algo(population_size=10, C_a=5, C_r=0.5, seed=self.seed)
        kh_griewankc = self.algo(population_size=10, C_a=5, C_r=0.5, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, kh_griewank, kh_griewankc)


class KHV2TestCase(AlgorithmTestCase):
    r"""Test case for KrillHerdV2 algorithm.

    Date:
        April 2019

    Author:
        Klemen Berkovič

    See Also:
        * :class:`niapy.tests.test_algorithm.AlgorithmTestCase`
    """

    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = KrillHerdV2

    def test_custom(self):
        kh_custom = self.algo(population_size=10, C_a=2, C_r=0.5, seed=self.seed)
        kh_customc = self.algo(population_size=10, C_a=2, C_r=0.5, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, kh_custom, kh_customc, MyBenchmark())

    def test_griewank(self):
        kh_griewank = self.algo(population_size=10, C_a=5, C_r=0.5, seed=self.seed)
        kh_griewankc = self.algo(population_size=10, C_a=5, C_r=0.5, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, kh_griewank, kh_griewankc)


class KHV3TestCase(AlgorithmTestCase):
    r"""Test case for KrillHerdV3 algorithm.

    Date:
        April 2019

    Author:
        Klemen Berkovič

    See Also:
        * :class:`niapy.tests.test_algorithm.AlgorithmTestCase`
    """

    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = KrillHerdV3

    def test_custom(self):
        kh_custom = self.algo(population_size=10, C_a=2, C_r=0.5, seed=self.seed)
        kh_customc = self.algo(population_size=10, C_a=2, C_r=0.5, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, kh_custom, kh_customc, MyBenchmark())

    def test_griewank(self):
        kh_griewank = self.algo(population_size=10, C_a=5, C_r=0.5, seed=self.seed)
        kh_griewankc = self.algo(population_size=10, C_a=5, C_r=0.5, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, kh_griewank, kh_griewankc)


class KHV4TestCase(AlgorithmTestCase):
    r"""Test case for KrillHerdV4 algorithm.

    Date:
        April 2019

    Author:
        Klemen Berkovič

    See Also:
        * :class:`niapy.tests.test_algorithm.AlgorithmTestCase`
    """

    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = KrillHerdV4

    def test_custom(self):
        kh_custom = self.algo(population_size=10, C_a=2, C_r=0.5, seed=self.seed)
        kh_customc = self.algo(population_size=10, C_a=2, C_r=0.5, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, kh_custom, kh_customc, MyBenchmark())

    def test_griewank(self):
        kh_griewank = self.algo(population_size=10, C_a=5, C_r=0.5, seed=self.seed)
        kh_griewankc = self.algo(population_size=10, C_a=5, C_r=0.5, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, kh_griewank, kh_griewankc)


class KHV11TestCase(AlgorithmTestCase):
    r"""Test case for KrillHerdV11 algorithm.

    Date:
        April 2019

    Author:
        Klemen Berkovič

    See Also:
        * :class:`niapy.tests.test_algorithm.AlgorithmTestCase`
    """

    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = KrillHerdV11

    def test_custom(self):
        kh_custom = self.algo(population_size=10, C_a=2, C_r=0.5, seed=self.seed)
        kh_customc = self.algo(population_size=10, C_a=2, C_r=0.5, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, kh_custom, kh_customc, MyBenchmark())

    def test_griewank(self):
        kh_griewank = self.algo(population_size=10, C_a=5, C_r=0.5, seed=self.seed)
        kh_griewankc = self.algo(population_size=10, C_a=5, C_r=0.5, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, kh_griewank, kh_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
