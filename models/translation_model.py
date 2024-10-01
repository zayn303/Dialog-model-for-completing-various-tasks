from transformers import pipeline
from classifier.task_classifier import TaskClassifier

class TranslationModel:
    def __init__(self):
        # Define models for each target language
        self.models = {
            "fr": pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr"),
            "de": pipeline("translation_en_to_de", model="Helsinki-NLP/opus-mt-en-de"),
            "it": pipeline("translation_en_to_it", model="Helsinki-NLP/opus-mt-en-it"),
            "sv": pipeline("translation_en_to_sv", model="Helsinki-NLP/opus-mt-en-sv"),
            "es": pipeline("translation_en_to_es", model="Helsinki-NLP/opus-mt-en-es"),
        }
        self.classifier = TaskClassifier()  

    def translate(self, text):
        # Classify the task to identify the target language
        task = self.classifier.classify_task(text)
        
        # Extract the target language based on the classification result
        target_lang = self.extract_target_language(task)

        # Raise an error if the target language is not supported
        if target_lang not in self.models:
            raise ValueError(f"Translation to {target_lang} is not supported.")
        
        # Return the translated text using the appropriate model
        return self.models[target_lang](text)[0]['translation_text']

    def extract_target_language(self, task):
        # Map the identified task to a target language
        language_mapping = {
            "translate to French": "fr",
            "translate to German": "de",
            "translate to Spanish": "es",
            "translate to Italian": "it",
            "translate to Swedish": "sv",
        }
        return language_mapping.get(task, None)
