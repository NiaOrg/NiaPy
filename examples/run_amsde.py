# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
# import sys
#
# sys.path.append('../')
# # End of fix
#
# from niapy.algorithms.basic import AgingNpMultiMutationDifferentialEvolution
# from niapy.algorithms.basic.de import cross_curr2best1, cross_best2
# from niapy.task import Task
# from niapy.problems import Sphere
#
# # we will run Differential Evolution for 5 independent runs
# for i in range(5):
#     task = Task(max_evals=5000, dimension=10, benchmark=Sphere())
#     algo = AgingNpMultiMutationDifferentialEvolution(population_size=10, F=0.2, CR=0.65,
#                                                      strategies=(cross_curr2best1, cross_best2), delta_np=0.05, omega=0.9)
#     best = algo.run(task)
#     print('%s -> %s' % (best[0], best[1]))
