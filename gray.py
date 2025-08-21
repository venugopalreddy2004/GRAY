from pymongo import MongoClient
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

class Gray:
    def __init__(self, connection_string: str, database: str, collection: str):
        self.client = MongoClient(connection_string)
        self.db = self.client[database]
        self.collection = self.db[collection]

    def store_model_details(
        self,
        model_name: str,
        hyperparameters: Dict[str, Any],
        performance_metrics: Dict[str, float],
        model_tags: List[str]
    ) -> str:
        document = {
            "model_name": model_name,
            "model_tags": model_tags,
            "hyperparameters": hyperparameters,
            "performance_metrics": performance_metrics,
            "timestamp": datetime.utcnow()
        }
        result = self.collection.insert_one(document)
        return str(result.inserted_id)

    def auto_store_model_details(
        self,
        model: BaseEstimator,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = None,
        model_tags: List[str] = [],
        metrics: list = ["accuracy", "f1_score"]
    ) -> str:
        if model_name is None:
            model_name = model.__class__.__name__

        hyperparameters = model.get_params()
        y_pred = model.predict(X_test)
        performance_metrics = {}

        if "accuracy" in metrics:
            performance_metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
        if "f1_score" in metrics:
            performance_metrics["f1_score"] = float(f1_score(y_test, y_pred, average='weighted'))

        return self.store_model_details(model_name, hyperparameters, performance_metrics, model_tags)

    def get_best_hyperparameters(
    self,
    model_name: str,
    metric: str = "accuracy",
    maximize: bool = True,
    model_tags: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        sort_order = -1 if maximize else 1

        candidates = list(self.collection.find({"model_name": model_name}))

        if not candidates:
            print("No models found with this name.")
            return None

        def score(doc):
            tags = set(doc.get("model_tags", []))
            tag_overlap = len(tags.intersection(model_tags)) if model_tags else 0
            tag_match_percent = tag_overlap / len(model_tags) if model_tags else 1.0

            perf = doc.get("performance_metrics", {}).get(metric, 0)
            perf_score = perf if maximize else (1 - perf)

            # Final score combines tag match and performance
            final_score = 0.5 * tag_match_percent + 0.5 * perf_score
            return final_score, tag_match_percent, perf_score

        best_doc = max(candidates, key=lambda doc: score(doc)[0])
        final_score, tag_match_percent, perf_score = score(best_doc)

        print(f"Recommended hyperparameters are {final_score * 100:.2f}% suitable.")
        print(f"Tag match: {tag_match_percent * 100:.2f}%, Model {metric}: {perf_score * 100:.2f}%")

        return {
            "hyperparameters": best_doc["hyperparameters"],
            "performance_metrics": best_doc["performance_metrics"],
            "model_tags": best_doc["model_tags"]
        }


    def __del__(self):
        self.client.close()