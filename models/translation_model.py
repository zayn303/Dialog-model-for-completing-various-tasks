from transformers import pipeline

class TranslationModel:
    def __init__(self, model_name="Helsinki-NLP/opus-mt-en-fr"):
        self.model = pipeline("translation_en_to_fr", model=model_name)

    def translate(self, text):
        return self.model(text)
