"""Event-driven family — DEFERRED. Registers no models and cannot trade.

Kept as a placeholder for the planned family (earnings drift, index adds, and
similar). Nothing here is wired up:

* there is no ``models.py``, so ``@register_model`` never runs for this family;
* ``meridian.families.__init__`` does not import it, so it is absent from the
  registry entirely — ``registry.list_families()`` returns **8** families, not 9;
* ``families/permissions.py`` maps it to ``_deferred``, which always returns
  ``False``, so even if a model appeared it would be blocked from trading.

:data:`ACTIVE` states this explicitly so the family count can be derived rather
than asserted. Docs and menus previously listed "nine strategy families" while
this one had never placed a trade, which overstated the platform's real coverage.

To activate: add ``models.py`` with at least one ``@register_model`` class, import
it from ``meridian/families/__init__.py``, give it a real entry in
``permissions.py``, and set ``ACTIVE = True``.
"""

#: This family is a stub. It registers no models and must not be counted as live.
ACTIVE = False

#: Why, in one line — surfaced in listings rather than left for a reader to infer.
STATUS = "deferred — stub only, no models registered, cannot trade"
