from transformers import pipeline

class TaskClassifier:
    def __init__(self):
        # Using a zero-shot classification model for task classification
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    def classify_task(self, input_text):
        # Define candidate labels for different tasks, including specific translations
        candidate_labels = [
            "translate to French", 
            "translate to German", 
            "translate to Spanish", 
            "translate to Italian", 
            "translate to Swedish",
            "summarization" 
            #"question answering", 
            #"classification"
        ]
        
        # Run zero-shot classification
        result = self.classifier(input_text, candidate_labels)
        
        # Extract the predicted task label
        task_label = result['labels'][0]  # Get the most likely label
        
        return task_label
