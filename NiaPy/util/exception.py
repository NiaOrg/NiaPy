# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, line-too-long, expression-not-assigned, len-as-condition, no-self-use, unused-argument, no-else-return, dangerous-default-value

class FesException(Exception):
	r"""Exception for exceeding number of maximum function evaluations.

	Author:
		Klemen Berkovič

	Date:
		2018

	License:
		MIT

	See Also:
		* :class:`Exception`
	"""
	def __init__(self, message='Reached the allowed number of the function evaluations!!!'):
		r"""Initialize the exception.

		Args:
			message (Optional[str]): Message show when this exception is thrown
		"""
		Exception.__init__(self, message)

class GenException(Exception):
	r"""Exception for exceeding number of algorithm iterations/generations.

	Author:
		Klemen Berkovič

	Date:
		2018

	License:
		MIT

	See Also:
		* :class:`Exception`
	"""
	def __init__(self, message='Reached the allowd number of the algorithm evaluations!!!'):
		r"""Initialize the exception.

		Args:
			message (Optional[str]): Message that is shown when this exceptions is thrown
		"""
		Exception.__init__(self, message)

class TimeException(Exception):
	r"""Exception for exceeding time limit.

	Author:
		Klemen Berkovič

	Date:
		2018

	License:
		MIT

	See Also:
		* :class:`Exception`
	"""
	def __init__(self, message='Reached the allowd run time of the algorithm'):
		r"""Initialize the exception.

		Args:
			message (Optional[str]): Message that is show when this exception is thrown.
		"""
		Exception.__init__(self, message)

class RefException(Exception):
	r"""Exception for exceeding reference value of function/fitness value.

	Author:
		Klemen Berkovič

	Date:
		2018

	License:
		MIT

	See Also:
		* :class:`Exception`
	"""
	def __init__(self, message='Reached the reference point!!!'):
		r"""Initialize the exception.

		Args:
			message (Optional[str]): Message that is show when this exception is thrown.
		"""
		Exception.__init__(self, message)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
