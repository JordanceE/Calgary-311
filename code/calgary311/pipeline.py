"""Top-level orchestration for the complete analytical workflow."""

from .closure_model import run_closure_analysis
from .community_clusters import run_community_clustering
from .config import PROCESSED, RANDOM_STATE, RAW, ensure_directories
from .reporting import generate_outputs
from .snow_weather import run_snow_weather_analysis


def run_analysis():
    """Run Q1–Q4 in sequence and regenerate all project outputs."""
    ensure_directories()
    context = {}
    context.update(run_snow_weather_analysis(RAW, PROCESSED, RANDOM_STATE))
    context.update(run_closure_analysis(RAW, PROCESSED, RANDOM_STATE))
    context.update(run_community_clustering(RAW, PROCESSED, RANDOM_STATE))
    results = generate_outputs(context)
    print("Analysis complete.")
    return results
