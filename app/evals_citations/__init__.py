"""Citation eval harness: score detailed_summary citations against GT.

Sibling to :mod:`app.evals` (the Ask / RAG harness) but focused on a
different pipeline: given a file's ``detailed_summary``, does
``app.citations.compute_citations`` pick source chunks that a human
curator would accept as the supporting evidence for each segment?

The harness deliberately reuses the snapshot / drive / runner
conventions from the Ask harness so the two read similarly. Cases are
YAML with per-segment expectations (chunk ids, or a time_range / page
hint). Metrics are computed segment-by-segment and aggregated per
case / per segment_type.
"""
