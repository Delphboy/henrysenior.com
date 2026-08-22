---
layout: archive
title: "Demos"
permalink: /demos/
author_profile: true
---

{% include base_path %}

Interactive demonstrations and visualisations exploring ideas from my research.

{% for post in site.demos %}
  {% include archive-single.html %}
{% endfor %}
