# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, multiple-statements, attribute-defined-outside-init, logging-not-lazy, arguments-differ, line-too-long, unused-argument
import logging
from numpy import apply_along_axis, argmin, argmax, log, exp, full
from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['HarmonySearch']

class HarmonySearch(Algorithm):
	r"""Implementation of harmony search algorithm.

	**Algorithm:** Harmony Search Algorithm

	**Date:** 2018

	**Authors:** Klemen Berkovič

	**License:** MIT

	**Reference URL:** https://link.springer.com/chapter/10.1007/978-3-642-00185-7_1

	**Reference paper:** Yang, Xin-She. "Harmony search as a metaheuristic algorithm." Music-inspired harmony search algorithm. Springer, Berlin, Heidelberg, 2009. 1-14.
	"""
	def __init__(self, **kwargs): Algorithm.__init__(self, name='HarmonySearch', sName='HS', **kwargs)

	def setParameters(self, HMS=30, r_accept=0.7, r_pa=0.35, b_range=1.42, **ukwargs):
		r"""Set the arguments of the algorithm.

		**Arguments:**

		HMS {integer} -- Number of harmonys in the memory

		r_accept {real} --

		r_pa {real} --

		b_range {real} --
		"""
		self.HMS, self.r_accept, self.r_pa, self.b_range = HMS, r_accept, r_pa, b_range
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def bw(self, task): return self.uniform(-1, 1) * self.b_range

	def adjustment(self, x, task): return x + self.bw(task)

	def improvize(self, HM, task):
		H = full(task.D, .0)
		for i in range(task.D):
			r, j = self.rand(), self.randint(self.HMS)
			H[i] = HM[j, i] if r > self.r_accept else self.adjustment(HM[j, i], task) if r > self.r_pa else self.uniform(task.Lower[i], task.Upper[i])
		return H

	def runTask(self, task):
		HM = self.uniform(task.Lower, task.Upper, [self.HMS, task.D])
		HM_f = apply_along_axis(task.eval, 1, HM)
		while not task.stopCondI():
			H = self.improvize(HM, task)
			H_f = task.eval(task.repair(H))
			iw = argmax(HM_f)
			if H_f <= HM_f[iw]: HM[iw], HM_f[iw] = H, H_f
		ib = argmin(HM_f)
		return HM[ib], HM_f[ib]

class HarmonySearchV1(HarmonySearch):
	r"""Implementation of harmony search algorithm.

	**Algorithm:** Harmony Search Algorithm

	**Date:** 2018

	**Authors:** Klemen Berkovič

	**License:** MIT

	**Reference URL:** https://link.springer.com/chapter/10.1007/978-3-642-00185-7_1

	**Reference paper:** Yang, Xin-She. "Harmony search as a metaheuristic algorithm." Music-inspired harmony search algorithm. Springer, Berlin, Heidelberg, 2009. 1-14.
	"""
	def setParameters(self, bw_min=1, bw_max=2, **kwargs):
		r"""Set the parameters of the algorithm.

		**Arguments:**

		bw_min {real} -- Minimal bandwidth

		bw_max {real} -- Maximal bandwidth
		"""
		self.bw_min, self.bw_max = bw_min, bw_max
		HarmonySearch.setParameters(self, **kwargs)

	def bw(self, task): return self.bw_min * exp(log(self.bw_min / self.bw_max) * task.Iters / task.nGEN)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
