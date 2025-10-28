from kedro.pipeline import Node, Pipeline  # noqa


from kedro.pipeline import Pipeline, node, pipeline
from .nodes import get_f1_sessions, load_sessions_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=get_f1_sessions,
            inputs=["params:cutoff_date", "params:start_year"],
            outputs="raw_f1_sessions",
            name="get_f1_sessions_node",
        ),
        node(
            func=load_sessions_data,
            inputs="raw_f1_sessions",
            outputs="loaded_f1_sessions",
            name="load_sessions_data_node",
        ),
    ])
