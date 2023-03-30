Releasing and Publishing TrueLearn
==================================

This document is intended for the core maintainers of the project.


Publishing to PyPI
------------------

.. important:: Make sure the secrets and tokens are setup correctly for the GitHub Actions.

When a new release is created the ``Upload Python Package`` action automatically deploys to
TestPyPi. This is to ensure that any issues can be identified before the official release.

After checking the package on TestPyPi run the upload python package workflow manually,
typing in ``pypi`` in the inputs box. TrueLearn will then be packaged and available on PyPi!


Release Process
---------------
The release process is as follows:

1. Create a new branch for the release named ``release-<version>``.
2. On the release branch, set the version number in the ``truelearn/__init__.py``
   removing the ``-dev`` suffix.
3. Update the ``CHANGELOG.rst`` file.
4. Commit the changes.
5. Open a pull request to merge into main.
6. Once approved, merge the pull request.
7. Then create a new tag for the release named ``<version>``.
8. Create a new release on GitHub with that tag with the title as ``TrueLearn <version>``
   and target the 'main' branch. The release notes should be
   copied from the ``CHANGELOG.rst`` file.
9. Read the Docs will build the documentation. The documentation will be
   available at http://truelearn.readthedocs.org/en/<version>/
   and the stable version will be updated to the latest release.

.. note:: You may need to activate the new version in the Read the Docs dashboard.

10. At this stage follow the instructions in the `Publishing to PyPI`_ section.

Post-release
------------
.. warning:: Only do this after the release has been published to PyPI. Otherwise
   the version number will be incorrect.

In the main branch update the version number in the ``truelearn/__init__.py`` file to the
next version number and add the ``-dev`` suffix. Also update the
``CHANGELOG.rst`` file to add a new section for the next version.
