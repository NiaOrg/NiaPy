================================
KNN Hyperparameter Optimization
================================

In this tutorial we will be using NiaPy to optimize the hyper-parameters of a KNN
classifier, using the Hybrid Bat Algorithm. We will be testing our implementation on
the UCI ML Breast Cancer Wisconsin (Diagnostic) dataset.

Dependencies
=============
Before we get started, make sure you have the following packages installed:

* **niapy**: :code:`pip install niapy --pre`
* **scikit-learn**: :code:`pip install scikit-learn`

Defining the problem
====================

Our problem consists of 4 variables for which we must find the most optimal
solution in order to maximize classification accuracy of K-nearest neighbors classifier.
Those variables are:

#. Number of neighbors (integer)
#. Weight function {'uniform', 'distance'}
#. Algorithm {‘ball_tree’, ‘kd_tree’, ‘brute’}
#. Leaf size (integer), used with the 'ball_tree' and 'kd_tree' algorithms

The solution will be a 4 dimensional vector with each variable representing a tunable
parameter of the KNN classifier. Since the problem variables in niapy are continuous real
values, we must map our solution vector :math:`\vec x; x_i \in [0, 1]` to integers:

* Number of neighbors: :math:`y_1 =  \lfloor 5 + x_1 \times 10 \rfloor; y_1 \in [5, 15]`
* Weight function: :math:`y_2 =  \lfloor x_2 \rceil; y_2 \in [0, 1]`
* Algorithm: :math:`y_3 =  \lfloor x_3 \times 2 \rfloor; y_3 \in [0, 2]`
* Leaf size: :math:`y_4 =  \lfloor 10 + x_4 \times 40 \rfloor; y_4 \in [10, 50]`

Implementation
==============

First we will implement two helper functions, which map our solution vector to the
parameters of the classifier, and construct said classifier.

.. code:: python

    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.neighbors import KNeighborsClassifier

    from niapy.problems import Problem
    from niapy.task import OptimizationType, Task
    from niapy.algorithms.modified import HybridBatAlgorithm


    def get_hyperparameters(x):
        """Get hyperparameters for solution `x`."""
        algorithms = ('ball_tree', 'kd_tree', 'brute')
        n_neighbors = int(5 + x[0] * 10)
        weights = 'uniform' if x[1] < 0.5 else 'distance'
        algorithm = algorithms[int(x[2] * 2)]
        leaf_size = int(10 + x[3] * 40)

        params =  {
            'n_neighbors': n_neighbors,
            'weights': weights,
            'algorithm': algorithm,
            'leaf_size': leaf_size
        }
        return params


    def get_classifier(x):
        """Get classifier from solution `x`."""
        params = get_hyperparameters(x)
        return KNeighborsClassifier(**params)

Next, we need to write a custom problem class. As discussed, the problem will be
4 dimensional, with lower and upper bounds set to 0 and 1 respectively. The class will
also store our training dataset, on which 2 fold cross validation will be performed.
The fitness function, which we'll be maximizing, will be the mean of the cross validation
scores.

.. code:: python

    class KNNHyperparameterOptimization(Problem):
        def __init__(self, X_train, y_train):
            super().__init__(dimension=4, lower=0, upper=1)
            self.X_train = X_train
            self.y_train = y_train

        def _evaluate(self, x):
            model = get_classifier(x)
            scores = cross_val_score(model, self.X_train, self.y_train, cv=2, n_jobs=-1)
            return scores.mean()

We will then load the breast cancer dataset, and split it into a train and test set
in a stratified fashion.

.. code:: python

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1234)

Now it's time to run the algorithm. We set the maximum number of iterations to 100,
and set the population size of the algorithm to 10.

.. code:: python

    problem = KNNHyperparameterOptimization(X_train, y_train)

    # We will be running maximization for 100 iters on `problem`
    task = Task(problem, max_iters=100, optimization_type=OptimizationType.MAXIMIZATION)

    algorithm = HybridBatAlgorithm(population_size=10, seed=1234)
    best_params, best_accuracy = algorithm.run(task)

    print('Best parameters:', get_hyperparameters(best_params))

Finally, let's compare our optimal model with the default one.

.. code:: python

    default_model = KNeighborsClassifier()
    best_model = get_classifier(best_params)

    default_model.fit(X_train, y_train)
    best_model.fit(X_train, y_train)

    default_score = default_model.score(X_test, y_test)
    best_score = best_model.score(X_test, y_test)

    print('Default model accuracy:', default_score)
    print('Best model accuracy:', best_score)

Output::

    Best parameters: {'n_neighbors': 8, 'weights': 'uniform', 'algorithm': 'kd_tree', 'leaf_size': 10}
    Default model accuracy: 0.9210526315789473
    Best model accuracy: 0.9385964912280702

