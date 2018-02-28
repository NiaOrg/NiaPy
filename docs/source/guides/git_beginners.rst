Git Beginners Guide
===================

Beginner's guide on how to contribute to open source community

.. note::

    If you don't have any previous experience with using Git, we recommend you take a 
    `15 minutes long Git Tutorial <https://try.github.io>`_.

Whether you're trying to give back to the open source community or collaborating on 
your own projects, knowing how to properly fork and generate pull requests is essential. 
Unfortunately, it's quite easy to make mistakes or not know what you should do when you're 
initially learning the process. I know that I certainly had considerable initial trouble 
with it, and I found a lot of the information on GitHub and around the internet to be 
rather piecemeal and incomplete - part of the process described here, another there, common 
hang-ups in a different place, and so on.

This short tutorial is fairly standard procedure for creating a fork, doing your work, 
issuing a pull request, and merging that pull request back into the original project.

Create a fork
-------------

Just head over to the our `GitHub page <https://github.com/NiaOrg/NiaPy>`_ and click the "Fork" 
button. It's just that simple. Once you've done that, you can use your favorite git client to 
clone your repo or just head straight to the command line:

.. code-block:: bash

    git clone git@github.com:<your-username>/<fork-project>


Keep your fork up to date
~~~~~~~~~~~~~~~~~~~~~~~~~

In most cases you'll probably want to make sure you keep your fork up to date by tracking the original 
"upstream" repo that you forked. To do this, you'll need to add a remote if not already added:

.. code-block:: bash

    # Add 'upstream' repo to list of remotes
    git remote add upstream git://github.com/NiaOrg/NiaPy.git


    # Verify the new remote named 'upstream'
    git remote -v

Whenever you want to update your fork with the latest upstream changes, you'll need to first fetch 
the upstream repo's branches and latest commits to bring them into your repository:

.. code-block:: bash

    # Fetch from upstream remote
    git fetch upstream

Now, checkout your own master branch and rebase with the upstream repo's master branch:

.. code-block:: bash

    # Checkout your master branch and merge upstream
    git checkout master
    git merge upstream/master

If there are no unique commits on the local master branch, git will simply perform a fast-forward. 
However, if you have been making changes on master (in the vast majority of cases you probably shouldn't 
be - see the next section :ref:`doing-your-work`, you may have to deal with conflicts. When doing so, be 
careful to respect the changes made upstream.

Now, your local master branch is up-to-date with everything modified upstream.

.. _doing-your-work:

Doing your work
---------------

Create a Branch
~~~~~~~~~~~~~~~

Whenever you begin work on a new feature or bug fix, it's important that you create a new branch. 
Not only is it proper git workflow, but it also keeps your changes organized and separated from the master 
branch so that you can easily submit and manage multiple pull requests for every task you complete.

To create a new branch and start working on it:

.. code-block:: bash

    # Checkout the master branch - you want your new branch to come from master
    git checkout master

    # Create a new branch named newfeature (give your branch its own simple informative name)
    git branch newfeature

    # Switch to your new branch
    git checkout newfeature

    # Last two commands can be joined as following: git checkout -b newfeature

Now, go to town hacking away and making whatever changes you want to

Submitting a Pull Request
-------------------------

Cleaning Up Your Work
~~~~~~~~~~~~~~~~~~~~~

Prior to submitting your pull request, you might want to do a few things to clean up your branch and 
make it as simple as possible for the original repo's maintainer to test, accept, and merge your work.

If any commits have been made to the upstream master branch, you should rebase your development branch 
so that merging it will be a simple fast-forward that won't require any conflict resolution work.

.. code-block:: bash

    # Fetch upstream master and merge with your repo's master branch
    git fetch upstream
    git checkout master
    git merge upstream/master

    # If there were any new commits, rebase your development branch
    git checkout newfeature
    git rebase master

Now, it may be desirable to squash some of your smaller commits down into a small number of larger more cohesive commits. You can do this with an interactive rebase:

.. code-block:: bash

    # Rebase all commits on your development branch
    git checkout 
    git rebase -i master

This will open up a text editor where you can specify which commits to squash.

Submitting
~~~~~~~~~~

Once you've committed and pushed all of your changes to GitHub, go to the page for your fork on GitHub, 
select your development branch, and click the pull request button. If you need to make any adjustments to 
your pull request, just push the updates to GitHub. Your pull request will automatically track the changes 
on your development branch and update.

When pull request is successfully created, make sure you follow activity on your pull request. It may occur 
that the maintainer of project will ask you to do some more changes or fix something on your pull request 
before merging it to master branch. 

After maintainer merges your pull request to master, you're done with development on this branch, so you're 
free to delete it.

.. code-block:: bash

    git branch -d newfeature

Copyright
---------

This guide is modified version of `original one <https://gist.github.com/Chaser324/ce0505fbed06b947d962>`_, 
written by Chase Pettit.

**Copyright**

Copyright 2017, Chase Pettit

`MIT License <http://www.opensource.org/licenses/mit-license.php>`_
 
**Additional Reading**

- `Atlassian - Merging vs. Rebasing <https://www.atlassian.com/git/tutorials/merging-vs-rebasing>`_

**Sources**

- `GitHub - Fork a Repo <https://help.github.com/articles/fork-a-repo>`_

- `GitHub - Syncing a Fork <https://help.github.com/articles/syncing-a-fork>`_

- `GitHub - Checking Out a Pull Request <https://help.github.com/articles/checking-out-pull-requests-locally>`_