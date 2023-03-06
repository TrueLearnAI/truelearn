Releasing and Publishing TrueLearn
==================================


This document is intended for the core maintainers of the project.

Publishing to PyPI
------------------
This is done automatically by GitHub Actions when a new release is created.
Make sure the secrets and tokens are set up correctly.

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
8. Create a new release on GitHub with the tag. The release notes should be
   copied from the ``CHANGELOG.rst`` file. The release title should be the
   version number and branch name should be the 'main' branch.
9. Read the Docs will build the documentation. The documentation will be
   available at http://truelearn.readthedocs.org/en/<version>/
   and the stable version will be updated to the latest release.

   *Note:* You may need to activate the new version in the Read the Docs dashboard.

Post-release
------------
In the main branch update the version number in the ``truelearn/__init__.py`` file to the
next version number and add the ``-dev`` suffix. Also update the
``CHANGELOG.rst`` file to add a new section for the next version.

Now development can continue.



