from models.translation_model import TranslationModel
from models.summarization_model import SummarizationModel

class TaskPipeline:
    def get_pipeline(self, task):
        # Initialize the translation and summarization models
        translation_model = TranslationModel()
        summarization_model = SummarizationModel()

        # Handle translation tasks specifically
        if "translate to" in task:
            # Directly use the translation method which requires the specific input
            return lambda input_text: translation_model.translate(input_text)
        elif task == "summarization":
            return summarization_model.summarize
        else:
            raise ValueError("Unknown task")

