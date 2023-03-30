:mod:`{{module}}`.{{objname}}
{{ underline }}==============

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('**Attributes**') }}
   {% for item in attributes %}
   .. autoattribute:: {{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('**Methods**') }}
   .. autosummary::
   {% for item in methods %}
       ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% if methods %}
   {% for method in methods %}
   .. automethod:: {{ method }}
   {% endfor %}
   {% endif %}

.. raw:: html

    <div class="clearer"></div>