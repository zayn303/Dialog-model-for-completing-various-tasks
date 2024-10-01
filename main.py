from classifier.task_classifier import TaskClassifier
from pipelines.task_pipelines import TaskPipeline

def main():
    classifier = TaskClassifier() 
    
    input_text = "Translate this text to Italian. There were thousands of flowers on a field"

    clean_up_tokenization_spaces = True

    # Classify the task
    task = classifier.classify_task(input_text)
    print(f"Task identified: {task}")

    # Get the task pipeline
    task_pipeline = TaskPipeline().get_pipeline(task)

    # Execute the pipeline with the input text
    result = task_pipeline(input_text)
    
    print(f"Result: {result}")

if __name__ == "__main__":
    main()
