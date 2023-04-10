:mod:`{{module}}`.{{objname}}
{{ underline }}==============

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}
   {% for item in attributes %}
   .. autoattribute:: {{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block methods %}
   {% if '__init__' in methods and methods|length == 1 %}
   {% else %}
   .. rubric:: {{ _('Methods') }}
   .. autosummary::
   {% for item in methods %}
   {% if item != '__init__' %}
       ~{{ name }}.{{ item }}
   {%- endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% if methods %}
   {% for method in methods %}
   {% if method != '__init__' %}
   .. automethod:: {{ method }}
   {% endif %}
   {% endfor %}
   {% endif %}
