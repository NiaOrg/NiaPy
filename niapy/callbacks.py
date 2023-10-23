class Callback:
    """Base class for callbacks.

    Callbacks allow you to execute code before and after each iteration of an algorithm.

    """

    def before_iteration(self, population, fitness, best_x, best_fitness, **params):
        """Callback method to be executed before each iteration of the algorithm.

        Args:
            population (numpy.ndarray): The current population of individuals.
            fitness (numpy.ndarray): The fitness values corresponding to the individuals.
            best_x (numpy.ndarray): The best solution found so far.
            best_fitness (float): The fitness value of the best solution found.
            **params: Additional algorithm parameters.

        """
        pass

    def after_iteration(self, population, fitness, best_x, best_fitness, **params):
        """Callback method to be executed after each iteration of the algorithm.

        Args:
            population (numpy.ndarray): The current population of individuals.
            fitness (numpy.ndarray): The fitness values corresponding to the individuals.
            best_x (numpy.ndarray): The best solution found so far.
            best_fitness (float): The fitness value of the best solution found.
            **params: Additional algorithm parameters.

        """
        pass


class CallbackList(Callback):
    """Container for Callback objects."""

    def __init__(self, callbacks=None):
        """Initialize CallbackList.

        Args:
            callbacks (list, optional): Existing list of callback objects. Defaults to None.

        """
        super().__init__()
        self.callbacks = list(callbacks) if callbacks else []

    def before_iteration(self, population, fitness, best_x, best_fitness, **params):
        """Execute before_iteration method for all callbacks.

        Args:
            population (numpy.ndarray): The current population of individuals.
            fitness (numpy.ndarray): The fitness values corresponding to the individuals.
            best_x (numpy.ndarray): The best solution found so far.
            best_fitness (float): The fitness value of the best solution found.
            **params: Additional algorithm parameters.

        """
        for callback in self.callbacks:
            callback.before_iteration(population, fitness, best_x, best_fitness, **params)

    def after_iteration(self, population, fitness, best_x, best_fitness, **params):
        """Execute after_iteration method for all callbacks.

        Args:
            population (numpy.ndarray): The current population of individuals.
            fitness (numpy.ndarray): The fitness values corresponding to the individuals.
            best_x (numpy.ndarray): The best solution found so far.
            best_fitness (float): The fitness value of the best solution found.
            **params: Additional algorithm parameters.

        """
        for callback in self.callbacks:
            callback.after_iteration(population, fitness, best_x, best_fitness, **params)

    def append(self, callback):
        """Append callback to list.

        Args:
            callback (Callback): Callback to append.

        Raises:
            ValueError: If callback is not an instance of `Callback`.

        """
        if isinstance(callback, Callback):
            self.callbacks.append(callback)
        else:
            raise ValueError('Callback must be an instance of `Callback`')
