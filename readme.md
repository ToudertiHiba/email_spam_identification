# Email Spam Identification Project

This project focuses on building a spam identification model. It involves data preprocessing, model training, and evaluation using various metrics.

## Overview

This project implements an email spam classifier using Natural Language Processing (NLP) techniques and a Multinomial Naive Bayes classifier. It aims to classify emails as either spam or legitimate (ham) based on their content.


## Files and Directories

- `email_spam_identification.py`: Python script containing the main code for the project.
- `email_spam_identification.ipynb`: Jupyter notebook version of the project.
- `spam.csv`: Dataset containing labeled email messages (ham or spam).
- `.venv_spam_identification`: Virtual environment directory.
- `pyproject.toml`: Poetry configuration files.
- `readme.md`: This file providing an overview of the project.


## Getting Started
1. Install Poetry for dependency management: [Poetry Installation Guide](https://python-poetry.org/docs/#installation).
2. Run `poetry install` to install the project dependencies.
3. Ensure you have the necessary dataset (`spam.csv`) in the project directory.
``


## Dependencies

- Python 3.x
- Pandas
- Scikit-learn


## Notes

- The main script `email_spam_identification.py` performs the entire process from loading data to evaluating the model.
- The virtual environment `.venv_spam_identification` is created using Poetry for managing dependencies.
- This project is a simplified example. In a real-world scenario, additional features, more advanced NLP techniques, and more sophisticated models might be used for improved accuracy.