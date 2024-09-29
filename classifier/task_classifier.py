from transformers import pipeline

class TaskClassifier:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english", token=None):
        
        self.classifier = pipeline("text-classification", model=model_name, use_auth_token=token)

    def classify_task(self, input_text):
        
        result = self.classifier(input_text)
        task_label = result[0]['label']
        return task_label
