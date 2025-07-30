
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

**Experiment Commands:**

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

# Runs the RAG experiment with summarized
python cli.py rag-summary

# Runs the RAG experiment with synthetic case
python cli.py rag-synthetic-cases

# Run all experiments
python cli.py all
```

**Synthetic Data Generation Commands:**

```bash
# Generate synthetic answers
python cli.py generate-synthetic-answers-cli

# Generate reasoning for the synthetic answers
python cli.py generate-reasoning-cli

# Summarize articles
python cli.py summarize-articles-cli

# Generate synthetic cases from articles
python cli.py generate-synthetic-cases-cli
```

**Customizing Models:**

You can easily override the default models using the `--model` and `--summarizer` options:

```bash
# Run the baseline experiment with a different model
python cli.py baseline --model="openai/gpt-4.1"

# Run the basic few-shot experiment with a different summarizer
python cli.py few-shot-basic --model="openai/gpt-4.1"
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