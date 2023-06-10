Development
===========

You can contribute to this package through its `GitHub Repository <https://github.com/NeLy-EPFL/flygym>`_.

Code style
----------
**Code:** We will use the `Black Code Style <https://black.readthedocs.io/en/stable/the_black_code_style/index.html>`_ (version 23.3.0). If you install FlyGym in the dev mode (`pip install -e ."[dev]"`), the Black formatter will be automatically installed. Please run `black . --check` in the root directory to check if your code is formatted correctly, or run `black .` to format all files. The GitHub Actions CI will also check the code style and display a red X if it is not compliant. You can also `integrate Black with your IDE <https://black.readthedocs.io/en/stable/integrations/index.html>`_. Comment lines should also be limited to 88 characters per line.

**Documentation:** We will use `Numpy Docstring Style <https://numpydoc.readthedocs.io/en/latest/format.html>`_ for docstrings and use a line length limit of 75 characters for docstrings.


Online documentation
--------------------

The online documentation is generated automatically using `Sphinx <https://www.sphinx-doc.org/en/master/>`_. The documentation is written in `reStructuredText <https://sphinx-tutorial.readthedocs.io/step-1/>`_ (RST). When you merge a pull request into the `main` branch, the documentation will be automatically built and deployed. If you want to check the documentation on a branch that is not `main` locally, you can run `make html` in the `doc` folder. The generated HTML files will be in `doc/build/html`. You can open `doc/build/html/index.html` in your browser to view the documentation.


API changes / migration guide
-----------------------------

Important API changes, including in particular non backward compatible API changes, should be documented on the :doc:`changelog` page (edit `doc/source/changelog.rst`).

.. For tutorial notebooks on Google Colab: Develop your notebook on Colab directly. Download the notebook as an ``.ipynb`` file and put it in the ``notebooks`` folder. Notebooks edited on Colab have a metadata block that tells Colab to, for example, use a GPU. If you edit the downloaded notebooks again locally, this block might be removed. Remember to add link to the documentation accordingly. Practically, I use the following steps to add a tutorial notebook:

.. #. Make necessary code changes in a new branch, eg. ``new-feature-branch``. Push it to the remote repository.
.. #. Write the new tutorial notebook with Colab. In the installation code block, specify the branch by appending ``@new-feature-branch`` to the GitHub URL in the ``pip install ...`` line. This line should now read ``!pip install -q --progress-bar=off 'flygym[mujoco] @ git+https://github.com/NeLy-EPFL/flygym.git@new-feature-branch'``.
.. #. Implement the rest of the tutorial notebook. When you're otherwise ready to merge, restart the notebook and run all cells from scratch. Remove the ``@new-feature-branch`` tag in the install block in the freshly run notebook. Download it to the ``notebooks`` directory, commit and push. The Colab notebook won't work now, but will work again when the code changes are merged into ``main``.
