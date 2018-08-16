# encoding=utf8
# pylint: disable=mixed-indentation, line-too-long, singleton-comparison, multiple-statements, attribute-defined-outside-init, no-self-use, logging-not-lazy, unused-variable, arguments-differ
import logging
from numpy import apply_along_axis, argmin, full, inf, where
from NiaPy.algorithms.algorithm import Algorithm
from NiaPy.benchmarks.utility import fullArray

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['ParticleSwarmAlgorithm']

class ParticleSwarmAlgorithm(Algorithm):
	r"""Implementation of Particle Swarm Optimization algorithm.

	**Algorithm:** Particle Swarm Optimization algorithm

	**Date:** 2018

	**Authors:** Lucija Brezočnik, Grega Vrbančič, Iztok Fister Jr. and Klemen Berkovič

	**License:** MIT

	**Reference paper:** Kennedy, J. and Eberhart, R. "Particle Swarm Optimization". Proceedings of IEEE International Conference on Neural Networks. IV. pp. 1942--1948, 1995.
	"""

	def __init__(self, **kwargs): Algorithm.__init__(self, name='ParticleSwarmAlgorithm', sName='PSO', **kwargs)

	def setParameters(self, NP=25, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, **ukwargs):
		r"""Set the parameters for the algorith.

		**Arguments:**

		NP {integer} -- population size

		C1 {decimal} -- cognitive component

		C2 {decimal} -- social component

		w {decimal} -- inertia weight

		vMin {decimal} -- minimal velocity

		vMax {decimal} -- maximal velocity
		"""
		self.NP, self.C1, self.C2, self.w, self.vMin, self.vMax = NP, C1, C2, w, vMin, vMax
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def init(self, task): self.w, self.vMin, self.vMax = fullArray(self.w, task.D), fullArray(self.vMin, task.D), fullArray(self.vMax, task.D)

	def repair(self, x, l, u):
		ir = where(x < l)
		x[ir] = l[ir]
		ir = where(x > u)
		x[ir] = u[ir]
		return x

	def runTask(self, task):
		"""Move particles in search space."""
		self.init(task)
		P, P_fit = task.Lower + task.bRange * self.rand([self.NP, task.D]), full(self.NP, inf)
		P_pb, P_pb_fit = P, P_fit
		p_b, p_b_fit = P[0], P_fit[0]
		V = full([self.NP, task.D], 0)
		while not task.stopCond():
			P = apply_along_axis(self.repair, 1, P, task.Lower, task.Upper)
			P_fit = apply_along_axis(task.eval, 1, P)
			ip_pb = where(P_pb_fit > P_fit)
			P_pb[ip_pb], P_pb_fit[ip_pb] = P[ip_pb], P_fit[ip_pb]
			ip_b = argmin(P_fit)
			if p_b_fit > P_fit[ip_b]: p_b, p_b_fit = P[ip_b], P_fit[ip_b]
			V = self.w * V + self.C1 * self.rand([self.NP, task.D]) * (P_pb - P) + self.C2 * self.rand([self.NP, task.D]) * (p_b - P)
			V = apply_along_axis(self.repair, 1, V, self.vMin, self.vMax)
			P = P + V
		return p_b, p_b_fit

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
