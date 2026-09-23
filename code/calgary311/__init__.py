"""Calgary 311 research pipeline, organized by analytical responsibility."""


def run_analysis():
    """Import the modeling stack only when the analysis is actually requested."""
    from .pipeline import run_analysis as _run_analysis

    return _run_analysis()


__all__ = ["run_analysis"]
