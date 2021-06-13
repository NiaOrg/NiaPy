===================================================
Feature selection using Particle Swarm Optimization
===================================================

In this tutorial we'll be using Particle Swarm Optimization to find an optimal subset of
features for a SVM classifier. We will be testing our implementation on the
UCI ML Breast Cancer Wisconsin (Diagnostic) dataset.

This tutorial is based on `Jx-WFST <https://github.com/JingweiToo/Wrapper-Feature-Selection-Toolbox>`_, a wrapper
feature selection toolbox, written in MATLAB by Jingwei Too.

Dependencies
============
Before we get started, make sure you have the following packages installed:

* **niapy**: :code:`pip install niapy --pre`
* **scikit-learn**: :code:`pip install scikit-learn`

Defining the problem
====================

We want to select a subset of relevant features for use in model construction, in
order to make prediction faster and more accurate. We will be using Particle Swarm
Optimization to search for the optimal subset of features.

Our solution vector will represent a subset of features:

.. math::

    x = [x_1, x_2, \dots , x_d]; x_i \in [0, 1]

Where :math:`d` is the total number of features in the dataset. We will then use
a threshold of 0.5 to determine whether the feature will be selected:

.. math::

    \\& x_i=
    \begin{cases}
      1, & \text{if}\ x_i > 0.5 \\
      0, & \text{otherwise}
    \end{cases}

The function we'll be optimizing is the classification accuracy penalized by the number
of features selected, that means we'll be minimizing the following function:

.. math::

    f(x) = \alpha \times (1 - P) + (1 - \alpha) \times \frac{N_selected}{N_features}

Where :math:`\alpha` is the parameter that decides the tradeoff between classifier
performance :math:`P` (classification accuracy in our case) and the number of selected
features with respect to the number of all features.

Implementation
==============

First we'll implement the Problem class, which implements the optimization function
defined above. It takes the training dataset, and the :math:`\alpha` parameter, which is
set to 0.99 by default.

For the objective function, the solution vector is first converted to binary, using the
threshold value of 0.5. That gives us indices of the selected features. If no features
were selected 1.0 is returned as the fitness. We then compute the mean accuracy of
running 2-fold cross validation on the training set, and calculate the value of the
optimization function defined above.

.. code:: python

    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.svm import SVC

    from niapy.problems import Problem
    from niapy.task import Task
    from niapy.algorithms.basic import ParticleSwarmOptimization


    class SVMFeatureSelection(Problem):
        def __init__(self, X_train, y_train, alpha=0.99):
            super().__init__(dimension=X_train.shape[1], lower=0, upper=1)
            self.X_train = X_train
            self.y_train = y_train
            self.alpha = alpha

        def _evaluate(self, x):
            selected = x > 0.5
            num_selected = selected.sum()
            if num_selected == 0:
                return 1.0
            accuracy = cross_val_score(SVC(), self.X_train[:, selected], self.y_train, cv=2, n_jobs=-1).mean()
            score = 1 - accuracy
            num_features = self.X_train.shape[1]
            return self.alpha * score + (1 - self.alpha) * (num_selected / num_features)

Then all we have left to do is load the dataset, run the algorithm and compare the results.

.. code:: python

    dataset = load_breast_cancer()
    X = dataset.data
    y = dataset.target
    feature_names = dataset.feature_names

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1234)

    problem = SVMFeatureSelection(X_train, y_train)
    task = Task(problem, max_iters=100)
    algorithm = ParticleSwarmOptimization(population_size=10, seed=1234)
    best_features, best_fitness = algorithm.run(task)

    selected_features = best_features > 0.5
    print('Number of selected features:', selected_features.sum())
    print('Selected features:', ', '.join(feature_names[selected_features].tolist()))

    model_selected = SVC()
    model_all = SVC()

    model_selected.fit(X_train[:, selected_features], y_train)
    print('Subset accuracy:', model_selected.score(X_test[:, selected_features], y_test))

    model_all.fit(X_train, y_train)
    print('All Features Accuracy:', model_all.score(X_test, y_test))

Output::

    Number of selected features: 4
    Selected features: mean smoothness, mean concavity, mean symmetry, worst area
    Subset accuracy: 0.9210526315789473
    All Features Accuracy: 0.9122807017543859

