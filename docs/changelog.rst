
Change Log
----------
This file documents all notable changes to the project.

- Date format is DD-MM-YYYY.
- TrueLearn adheres to `Semantic Versioning`_.
- TrueLearn uses the `Keep a Changelog`_ format.

.. _Semantic Versioning: http://semver.org/
.. _Keep a Changelog: http://keepachangelog.com/


**TrueLearn 1.1.0 (13-08-2023)**

*Added*

- The option to visualise the variance in Bar, Dot, Line plotters. By default this is
  enabled except in the Line plotter.

*Changed*

- INK classifier now uses a binary skill representation.
- The INK Classifier can now be instantiated using a dictionary of parameters
- Added black border to Rose Plotter to make it easier to distinguish topics of similar
  colour.
- Locked the plotly, kaleido and wordcloud dependencies to avoid breaking changes.
- Radar plotter doesnt show the variance, due to concerns about the representation of
  variance being misleading.

*Fixed*

- Incorrect weights used in INK prediction
- Topic label overflow in Bubble plotter. This means that a reasonable number
  of topics (around 10 to 15) can be plotted for the default image size.
- Wordcloud now fully supports Python 3.11

*Removed*

- orjson dependency, as JSON deserialisation is not our bottleneck and the additional dependency
  is not worth it.

**TrueLearn 1.0.0 (14-04-2023)**

- Initial release!
