Contributing
============

You can contribute to this package through its `GitHub Repository <https://github.com/NeLy-EPFL/flygym>`_.


Code of conduct & licensing
---------------------------
Please respect the `Contributor Covenant Code of Conduct <https://www.contributor-covenant.org/version/2/1/code_of_conduct/code_of_conduct.txt>`_. FlyGym is made open source under `Apache License 2.0 <https://github.com/NeLy-EPFL/flygym/blob/main/LICENSE>`_. By contributing to this package (including any issue, pull request, and discussion), you agree that your content will be shared under the same license.


Branches
--------
- ``main``: The latest stable code. Every time this branch is updated (except documentation-only updates to the neuromechfly.org website), a new version should be pushed to PyPI.
- ``develop``: The latest development code. This branch is used for development and testing. Code should not be merged into this branch until all tests and style checks are passing. Contribution from developers outside the core dev team (Sibo Wang-Chen and Victor Alfred Stimpfling) should have their PRs reviewed by a core dev team member. When a new version is released, the ``develop`` branch is merged into the ``main`` branch.
- **Other branches** are for develop new features or fixing bugs. Please make your own fork for development (see the "Contributing to the codebase" section below).


Code style
----------
**Code:** We use the `Black Code Style <https://black.readthedocs.io/en/stable/the_black_code_style/index.html>`_ (version 23.3.0). If you install FlyGym in the dev mode (``pip install -e ."[dev]"``), the Black formatter will be automatically installed. Please run ``black . --check`` in the root directory to check if your code is formatted correctly, or run ``black .`` to format all files. When you push your code, GitHub will check the code style and display a red X if it is not compliant — this prevents you from merging your code into the main branch. You can also `integrate Black with your IDE <https://black.readthedocs.io/en/stable/integrations/index.html>`_. Comment lines should also be limited to 88 characters per line.


Documentation
-------------
We use the  `NumPy Docstring Style <https://numpydoc.readthedocs.io/en/latest/format.html>`_. We use a line length limit of 75 characters for docstrings. Please stick with the NumPy style so the API reference can be generated automatically.

The source files (in RST) of the documentation website are located in the ``doc/source`` folder. The API reference is generated automatically using `Sphinx <https://www.sphinx-doc.org/en/master/>`_. The documentation is written in `reStructuredText <https://sphinx-tutorial.readthedocs.io/step-1/>`_ (RST). When you merge a pull request into the main branch, the documentation is automatically built and deployed on `neuromechfly.org <https://neuromechfly.org/>`_. If you want to check the documentation on a branch (that is not `main`) locally, you can run `make html` under the `doc` folder. The generated HTML files will be placed under `doc/build/html`. You can open `doc/build/html/index.html` in your browser to view the documentation.

API changes / migration guide
-----------------------------

Important API changes, including in particular non backward compatible API changes, should be documented on the `Change Log <https://neuromechfly.org/changelog.html>`_ page (edit ``doc/source/changelog.rst``).


Contributing to the codebase
----------------------------

.. note::
   The following content is adapted from the `SLEAP Contributing Guide <https://github.com/talmolab/sleap/blob/develop/docs/CONTRIBUTING.md>`_ (BSD License).

1. Install the package using the developer mode from the `develop branch <https://github.com/NeLy-EPFL/flygym/tree/develop>`_.
2. Create a fork from the ``develop`` branch.

   - Either work on the develop branch or create a new branch (recommended if tackling multiple issues at a time).
   - If creating a branch, use your name followed by a relevant keyword for your changes, eg: ``git checkout -b john/some_issue``

3. Make some changes/additions to the source code that tackle the issue(s).
4. Write `tests <https://github.com/NeLy-EPFL/flygym/tree/main/flygym/mujoco/tests>`_.

   - You can either write tests before creating a draft pull request (PR), or submit draft PR (to get code coverage statistics via codecov) and then write tests to narrow down error prone lines.
   - Test(s) should aim to hit every point in the proposed change(s) - cover edge cases to best of your ability.
   - Try to hit code coverage points.
   - Add files, commit, and push to origin.

5. Create a draft PR on Github.

   - Make sure the tests pass and code coverage is good.

6. If either the tests or style checks fail, repeat steps 3-5.
7. Once the draft PR looks good, convert to a finalized PR (hit the ready for review button).

   - IMPORTANT: Only convert to a finalized PR when you believe your changes are ready to be merged.
   - Optionally assign a reviewer on the right of the screen - otherwise a member of the developer team will self-assign themselves.

8. If the reviewer requests changes, repeat steps 3-5 and re-request review.
9. Once the reviewer signs off they will squash + merge the PR into the ``develop`` branch.

   - New features will be available on the main branch when a new version is released.
