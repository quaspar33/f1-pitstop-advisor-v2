from kedro.pipeline import Pipeline, node, pipeline
from .nodes import aggregate_laps_by_circuit


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=aggregate_laps_by_circuit,
                inputs=["loaded_sessions", "compounds_map"],
                outputs="circuit_lap_data",
                name="aggregate_laps_by_circuit_node",
            ),
        ]
    )
