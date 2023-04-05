Development
===========

You can contribute to this package through its `GitHub Repository <https://github.com/NeLy-EPFL/flygym>`_.

This package is used for a course. For stability issue, please do not merge directly to main but create a pull request instead. If you want to change the existing API, please discuss with us or create an issue first.

For documentation:

* For pages on the documentation website, edit the corresponding `RST files <https://sphinx-tutorial.readthedocs.io/step-1/>`_ in the ``doc`` folder. RST is like Markdown, but more suitable for technical documentation.
* For code docstrings, please use the `NumPy documentation style <https://numpydoc.readthedocs.io/en/latest/format.html>`_; this will allow API docs to be generated automatically.
* For existing classes/functions, modifying the docstrings in the Python modules will be sufficient. The API reference will be updated automatically. For new classes/functions: you need to add pages under ``doc`` accordingly.

For tutorial notebooks on Google Colab: Develop your notebook on Colab directly. Download the notebook as an ``.ipynb`` file and put it in the ``notebooks`` folder. Notebooks edited on Colab have a metadata block that tells Colab to, for example, use a GPU. If you edit the downloaded notebooks again locally, this block might be removed. Remember to add link to the documentation accordingly.