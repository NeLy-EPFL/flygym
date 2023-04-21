Development
===========

You can contribute to this package through its `GitHub Repository <https://github.com/NeLy-EPFL/flygym>`_.

This package is used for a `course <https://github.com/NeLy-EPFL/cobar-miniproject-2023/wiki>`_. For stability purposes, please do not merge directly to the main branch (really minor changes or docs changes excepted). Create a pull request instead. If you want to change the existing API, please discuss with Sibo or create an issue first.

For documentation:

* For pages on the documentation website, edit the corresponding `RST files <https://sphinx-tutorial.readthedocs.io/step-1/>`_ in the ``doc/source`` folder. RST is like Markdown, but more suitable for technical documentation.
* For code docstrings, please use the `NumPy documentation style <https://numpydoc.readthedocs.io/en/latest/format.html>`_; this will allow API docs to be generated automatically.
* To edit the API reference for existing classes or functions, modifying the docstrings in the Python modules will be sufficient. The API reference will be updated automatically. For new classes for functions: you need to add pages under ``doc`` accordingly.

Important API changes, including in particular non backward compatible API changes, should be documented on the :doc:`changelog` page (edit `doc/source/changelog.rst`).

For tutorial notebooks on Google Colab: Develop your notebook on Colab directly. Download the notebook as an ``.ipynb`` file and put it in the ``notebooks`` folder. Notebooks edited on Colab have a metadata block that tells Colab to, for example, use a GPU. If you edit the downloaded notebooks again locally, this block might be removed. Remember to add link to the documentation accordingly. Practically, I use the following steps to add a tutorial notebook:

#. Make necessary code changes in a new branch, eg. ``new-feature-branch``. Push it to the remote repository.
#. Write the new tutorial notebook with Colab. In the installation code block, specify the branch by appending ``@new-feature-branch`` to the GitHub URL in the ``pip install ...`` line. This line should now read ``!pip install -q --progress-bar=off 'flygym[mujoco] @ git+https://github.com/NeLy-EPFL/flygym.git@new-feature-branch'``.
#. Implement the rest of the tutorial notebook. When you're otherwise ready to merge, restart the notebook and run all cells from scratch. Remove the ``@new-feature-branch`` tag in the install block in the freshly run notebook. Download it to the ``notebooks`` directory, commit and push. The Colab notebook won't work now, but will work again when the code changes are merged into ``main``.
