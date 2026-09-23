# Zepto Data & AI Platform

A capstone project covering data engineering, analytics and machine learning, and an AI-powered support assistant.

## Project Structure

- data_pipeline/
- analytics/
- support_assistant/
- requirements.txt
- README.md

## Module 1 - Data Pipeline

The pipeline scrapes book data from Books to Scrape for Travel, Mystery, and Historical Fiction categories.

It stores the data in SQLite using categories and books tables with a foreign-key relationship.

Main files:

- data_pipeline/output/books.db
- data_pipeline/queries.sql

## Module 2 - Analytics and Machine Learning

The analytics module uses the Titanic dataset and performs exploratory analysis, preprocessing, classification, class-imbalance handling, Random Forest tuning, and regression analysis.

Models include:

- Logistic Regression
- Decision Tree
- Random Forest

The final fitted preprocessing and model pipeline is saved as:

analytics/best_titanic_pipeline.joblib

The offline dataset is:

analytics/titanic.csv

## Module 3 - AI Support Assistant

The support assistant uses eight policy documents, Sentence Transformers embeddings, ChromaDB, and LangGraph.

It provides a FastAPI /ask endpoint and supports local/offline execution using deterministic mock responses.

Main files:

- support_assistant/app.py
- support_assistant/Dockerfile
- support_assistant/docs/

## Installation

Install the required packages with:

    pip install -r requirements.txt

## Module 3 API

From the support_assistant directory:

    uvicorn app:app --host 0.0.0.0 --port 8000

Docker:

    docker build -t zepto-support-assistant .
    docker run -p 8000:8000 zepto-support-assistant

## Design Decisions

SQLite is used for lightweight relational storage.

Pandas is used for data transformation and validation.

Scikit-learn pipelines combine preprocessing and machine learning.

Stratified splitting is used for classification evaluation.

ChromaDB provides persistent local vector storage.

Sentence Transformers provides local document embeddings.

LangGraph provides explicit workflow routing.

The project is designed to run locally without requiring paid services.

## Reproducibility

The repository contains the datasets, generated outputs, model artifacts, support documents, and configuration required to reproduce the project workflows.


## Git Workflow

This project was developed using a feature-branch workflow. The completed platform was committed on a dedicated feature branch and then merged into the main branch.
The repository contains the complete data pipeline, analytics, and support assistant modules.
