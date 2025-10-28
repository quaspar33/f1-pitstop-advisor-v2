from typing import Dict
from kedro.pipeline import Pipeline
from f1_pitstop_advisor.pipelines import data_ingestion, feature_engineering


def register_pipelines() -> Dict[str, Pipeline]:
    data_ingestion_pipeline = data_ingestion.create_pipeline()
    feature_engineering_pipeline = feature_engineering.create_pipeline()

    return {
        "__default__": data_ingestion_pipeline + feature_engineering_pipeline,
        "data_ingestion": data_ingestion_pipeline,
        "feature_engineering": feature_engineering_pipeline,
    }
