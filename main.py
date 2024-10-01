from classifier.task_classifier import TaskClassifier
from pipelines.task_pipelines import TaskPipeline
import warnings

# Ignore the FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

def main():
    # Welcome message
    print("Welcome to Dialog Model for Completing Translation and Summarizing Tasks!")

    # Prompt the user to enter text
    input_text = input("Please enter the text: ")

    # Initialize the task classifier
    classifier = TaskClassifier()

    # Classify the task based on user input
    task = classifier.classify_task(input_text)
    print(f"Task identified: {task}")

    # Get the appropriate pipeline for the identified task
    task_pipeline = TaskPipeline().get_pipeline(task)

    # Execute the pipeline with the user's input text
    result = task_pipeline(input_text)
    
    # If the task is summarization, extract the summary text
    if task == "summarization":
        # Assuming the result is a list of dictionaries, extract the 'summary_text'
        summary = result[0]['summary_text']
        print(f"Result: {summary}")
    else:
        # For translation, output the result directly
        print(f"Result: {result}")

if __name__ == "__main__":
    main()
