from transformers import pipeline

class SummarizationModel:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        self.model = pipeline("summarization", model=model_name)

    def summarize(self, text):
        return self.model(text)
