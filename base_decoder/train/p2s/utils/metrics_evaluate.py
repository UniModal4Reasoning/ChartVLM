from typing import Any
import evaluate

class Metrics(object):
    def __init__(self, metrics_name="exact_match"):
        
        assert metrics_name in ["rouge", "exact_match", "relaxed_acc", "csv_metric"]
        self.metrics_name = metrics_name
        
        if self.metrics_name == "rouge":        
            self.metrics = evaluate.load("")
        elif self.metrics_name == "exact_match":
            self.metrics = evaluate.load("")
        elif self.metrics_name == "relaxed_acc":
            self.metrics = evaluate.load("")
        elif self.metrics_name == "csv_metric":
            self.metrics = evaluate.load("")
        else:
            raise NotImplementedError
        
    def __call__(self, predictions, references):
        result = self.metrics.compute(
            predictions=predictions,
            references=references
        )
        
        if isinstance(result, dict):
            result = list(result.values())[0]
            
        return result
        