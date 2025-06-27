# Dialog Model for Completing Various Tasks

This project provides a modular dialog-based system capable of performing multiple NLP tasks such as summarization, translation, and classification. It is structured into reusable components, making it easy to extend and maintain.

## Features

- Task classification using pretrained transformer models  
- Summarization pipeline  
- Translation pipeline  
- Easily extendable architecture  
- CLI-based interface via `main.py`

## Project Structure

```
.
├── main.py                         # Entry point for running the system
├── requirements.txt               # Python dependencies
├── classifier/                    # Task classifier logic
│   └── task_classifier.py
├── models/                        # Task-specific models (summarization, translation)
│   ├── summarization_model.py
│   └── translation_model.py
├── pipelines/                     # Processing pipelines
├── utils/                         # Utility functions
└── .gitignore
```

## Installation

1. Clone the repository:

```
git clone https://github.com/zayn303/dialog-model-for-completing-various-tasks.git
cd dialog-model-for-completing-various-tasks
```

2. Create a virtual environment and install dependencies:

```
python3 -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run the main script with your input text:

```
python main.py --text "Translate this sentence." --language "en"
```

The system will automatically detect the task and execute the corresponding pipeline.

## Tasks Supported

- Summarization (e.g., news, long articles)  
- Translation (English to other languages)  
- Classification (intent recognition)

## Author

Developed by Andrii Kolomiiets as part of an academic project.

## License

This project is licensed under the MIT License.
