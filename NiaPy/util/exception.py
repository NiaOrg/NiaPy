# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, line-too-long, expression-not-assigned, len-as-condition, no-self-use, unused-argument, no-else-return, old-style-class, dangerous-default-value

class FesException(Exception):
	def __init__(self, message='Reached the allowd number of the function evaluations!!!'): Exception.__init__(self, message)

class GenException(Exception):
	def __init__(self, message='Reached the allowd number of the algorithm evaluations!!!'): Exception.__init__(self, message)

class TimeException(Exception):
	def __init__(self, message='Reached the allowd run time of the algorithm'): Exception.__init__(self, message)

class RefException(Exception):
	def __init__(self, message='Reached the reference point!!!'): Exception.__init__(self, message)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
