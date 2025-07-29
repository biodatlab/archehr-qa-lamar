# Clinical Note Question Answering System

This project is a flexible and powerful system for running question-answering experiments on clinical notes. It provides a modular framework for testing different approaches, including baseline, few-shot, and Retrieval-Augmented Generation (RAG) methods. The system is designed to be highly configurable, allowing you to easily swap out models from any OpenAI-compatible API, such as OpenRouter.

## Features

- **Multiple Experiment Types**: Run baseline, few-shot, and RAG experiments out of the box.
- **Configurable Models**: Easily configure the language model for each experiment.
- **Jupyter Notebook Interface**: An interactive `main.py` notebook for easy experimentation and visualization.
- **Command-Line Interface**: A powerful CLI for automating experiment runs.
- **Modular and Extensible**: The project is designed with a clean and modular structure, making it easy to extend and adapt.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1.  **Create a `.env` file** in the root of the project. You can copy the example file:
    ```bash
    cp .env.example .env
    ```

2.  **Add your API key** to the `.env` file:
    ```
    OPENAI_API_KEY="YOUR_API_KEY"
    ```

3.  **If you are using a service other than OpenAI (e.g., OpenRouter), add the base URL to your `.env` file:**
    ```
    OPENAI_BASE_URL="https://openrouter.ai/api/v1"
    ```

## Usage

You can run the experiments using either the Jupyter-style notebook (`main.py`) or the command-line interface (`cli.py`).

### Notebook Interface

The `main.py` file is structured like a Jupyter notebook, with code cells separated by `# %%`. You can run it in any editor that supports this format (e.g., VS Code, PyCharm).

1.  **Open `main.py`** in your editor.
2.  **Configure the models** at the top of the file:
    ```python
    BASELINE_MODEL = "google/gemini-2.5-pro"
    FEWSHOT_MODEL = "google/gemini-2.0-flash"
    RAG_MODEL = "google/gemini-2.5-pro"
    SUMMARIZATION_MODEL = "google/gemini-2.5-pro"
    ```
3.  **Run the cells** individually to execute each step of the experiments.

### Command-Line Interface

The `cli.py` script provides a powerful way to run experiments from the command line.

**Basic Commands:**

```bash
# Run the baseline experiment
python cli.py baseline

# Run the basic few-shot experiment
python cli.py few-shot-basic

# Run the few-shot experiment with synthetic answers
python cli.py few-shot-syn-ans

# Run the few-shot experiment with synthetic answers and reasoning
python cli.py few-shot-syn-w-reasoning

# Run the RAG experiment
python cli.py rag

# Run all experiments
python cli.py all
```

**Customizing Models:**

You can easily override the default models using the `--model` and `--summarizer` options:

```bash
# Run the baseline experiment with a different model
python cli.py baseline --model="anthropic/claude-3-haiku"

# Run the basic few-shot experiment with a different summarizer
python cli.py few-shot-basic --summarizer="anthropic/claude-3-sonnet"
```

## Project Structure

```
.
├── cli.py                  # Command-line interface
├── config.py               # Project configuration
├── data/                     # Raw data files
├── data_processing/        # Data loading and parsing scripts
├── evaluation/             # Evaluation metrics
├── generation/             # Prompt and answer generation scripts
├── main.py                 # Main notebook for experiments
├── output/                 # Generated output files
├── postprocess.py          # Post-processing scripts
├── prompts/                # Prompt templates
├── rag/                    # Retrieval-Augmented Generation scripts
├── requirements.txt        # Project dependencies
├── utils/                  # Utility functions
└── README.md               # This file
``` 