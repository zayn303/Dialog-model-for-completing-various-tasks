from classifier.task_classifier import TaskClassifier
from pipelines.task_pipelines import TaskPipeline
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main():
    api_key = "API_KEY"
    
    classifier = TaskClassifier(token=api_key)
    
    input_text = "Translate this text to French."

    task = classifier.classify_task(input_text)
    print(f"Task identified: {task}")

    task_pipeline = TaskPipeline().get_pipeline(task)

    result = task_pipeline(input_text)
    print(f"Result: {result}")

if __name__ == "__main__":
    main()
