# encoding=utf8

"""Python micro framework for building nature-inspired algorithms."""

from niapy import util, algorithms, problems, task
from niapy.runner import Runner

__all__ = ["algorithms", "problems", "util", "task", "Runner"]
__project__ = "NiaPy"
__version__ = "2.0.0rc17"

VERSION = "{0} v{1}".format(__project__, __version__)
