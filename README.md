
# NLP Models for Ancient Greek

This repository contains the entire codebase for the B.Sc. thesis focused on Natural Language Processing (NLP) for the Ancient Greek language. The original thesis paper can be reviewed [here](https://pergamos.lib.uoa.gr/uoa/dl/frontend/en/browse/3100154).

We have developed Language Models (LMs) on the back of the rising transformer Neural Network technology for Ancient Greek language, a language that has surprisingly seen little presence in the field of LMs. Our models surpass previous baselines and existing models in performance when fine-tuned on downstream tasks.

## Getting Started

We strongly recommend using a Unix-like OS for this project, due to the nature of the scripts. Follow the steps below to set up:

1. Install [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

2. Clone the repo using the command:
    ```shell
    git clone https://github.com/Simplicity-Sprint/NLP-AncientGreek-Thesis.git && cd NLP-AncientGreek-Thesis
    ```

3. Create and activate a virtual environment using the command:
    ```shell
    conda create --name ag-nlp-venv python=3.8 && conda activate ag-nlp-venv
    ```

4. Install the required packages using the command:
    ```shell
    pip install -r requirements.txt
    ```

You can now follow the rest of the instructions available in the repository regarding data download, training, and hyperparameter tuning.

## Contribution

Feel free to contribute to the code, or raise issues you come across. Let's work together to advance NLP for Ancient Greek!