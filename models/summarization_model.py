from transformers import pipeline

class SummarizationModel:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        # Create the summarization pipeline
        self.model = pipeline("summarization", model=model_name)

    def summarize(self, text):
        # Pass clean_up_tokenization_spaces=False to avoid the warning
        return self.model(text, clean_up_tokenization_spaces=False)
