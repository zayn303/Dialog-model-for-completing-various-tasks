from models.translation_model import TranslationModel
from models.summarization_model import SummarizationModel

class TaskPipeline:
    def get_pipeline(self, task):
        if task == "translation":
            return TranslationModel().translate
        elif task == "summarization":
            return SummarizationModel().summarize
        else:
            raise ValueError("Unknown task")
