:mod:`{{module}}`.{{objname}}
{{ underline }}==============

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   .. rubric:: {{ _('Methods') }}
   .. automethod:: __call__
   {% endblock %}

.. raw:: html

    <div class="clearer"></div>

