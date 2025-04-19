# Home-Assignment 4

CS5720 Home Assignment 4

Student Name: Dama Vamsi

Student ID: 700771673

Course: Neural Networks and Deep Learning (CS5720)

Semester: Spring 2025

University: University of Central Missouri

Department: Computer Science 

Overview
This repository contains the solutions for Home Assignment 4, covering Chapters 9 and 10 of the CS5720 course. The assignment includes four tasks focused on Natural Language Processing (NLP) and neural network concepts:

Q1: NLP preprocessing pipeline (tokenization, stopword removal, stemming).
Q2: Named Entity Recognition (NER) using SpaCy.
Q3: Scaled dot-product attention mechanism implementation.
Q4: Sentiment analysis using HuggingFace Transformers.

Each task is implemented in a separate Python script, with appropriate comments and error handling to ensure robust execution. This README provides instructions to set up and run the code, along with brief explanations of each solution.
Repository Structure

q1_nlp_preprocessing.py: Implements the NLP preprocessing pipeline.
q2_ner_spacy.py: Performs NER using SpaCy.
q3_attention.py: Computes scaled dot-product attention.
q4_sentiment.py: Conducts sentiment analysis with HuggingFace.
README.md: This file, explaining the project and instructions.

Requirements
To run the code, you need the following:

Python Version: 3.8 or higher
Libraries:
nltk (for Q1)
spacy (for Q2)
numpy, scipy (for Q3)
transformers (for Q4)


SpaCy Model: en_core_web_sm (for Q2)
NLTK Resources: punkt, punkt_tab, stopwords (for Q1)

Install the dependencies using:
pip install nltk spacy numpy scipy transformers
python -m spacy download en_core_web_sm

For NLTK resources, the scripts automatically download punkt, punkt_tab, and stopwords when executed.
Setup Instructions

Clone the Repository:
git clone <your-repo-url>
cd <repo-directory>


Install Dependencies:Run the pip command above to install all required libraries.

Google Colab Setup (if using Colab):

Upload all .py files to Colab’s /content directory.
Run the following in a Colab cell to install dependencies:!pip install nltk spacy numpy scipy transformers
!python -m spacy download en_core_web_sm


The scripts handle NLTK resource downloads automatically.


Verify SpaCy Model:Ensure en_core_web_sm is installed. If not, the Q2 script will attempt to download it automatically.


Running the Code
Each script can be run independently. Use the following commands:

Q1: NLP Preprocessing:
python q1_nlp_preprocessing.py


Input: Sentence: "NLP techniques are used in virtual assistants like Alexa and Siri."
Output: Prints original tokens, tokens without stopwords, and stemmed words.


Q2: Named Entity Recognition:
python q2_ner_spacy.py


Input: Sentence: "Barack Obama served as the 44th President of the United States and won the Nobel Peace Prize in 2009."
Output: Prints entity text, label, and character positions.


Q3: Scaled Dot-Product Attention:
python q3_attention.py


Input: Matrices Q, K, V (predefined in the script).
Output: Prints attention weights and final output matrix.


Q4: Sentiment Analysis:
python q4_sentiment.py


Input: Sentence: "Despite the high price, the performance of the new MacBook is outstanding."
Output: Prints sentiment label and confidence score.



In Google Colab, run each script using:
!python q1_nlp_preprocessing.py
!python q2_ner_spacy.py
!python q3_attention.py
!python q4_sentiment.py

Solution Explanations

Q1: NLP Preprocessing Pipeline:

Uses NLTK to tokenize a sentence, remove English stopwords, and apply Porter stemming.
The script includes error handling to download required NLTK resources (punkt, punkt_tab, stopwords) if missing.
Example output: Shows tokenized words, filtered tokens, and stemmed forms.


Q2: Named Entity Recognition with SpaCy:

Employs SpaCy’s en_core_web_sm model to extract named entities (e.g., PERSON, GPE) from a sentence.
Automatically installs the SpaCy model if not present, with error handling for failed installations.
Output lists entities, their labels, and character positions.


Q3: Scaled Dot-Product Attention:

Implements the attention mechanism using NumPy and SciPy’s softmax.
Takes Query (Q), Key (K), and Value (V) matrices, computes scaled dot-product attention, and outputs attention weights and the final matrix.
No external dependencies beyond NumPy and SciPy.


Q4: Sentiment Analysis using HuggingFace:

Uses HuggingFace’s transformers library with the distilbert-base-uncased-finetuned-sst-2-english model for sentiment analysis.
Suppresses TensorFlow warnings and includes error handling for robust execution.
Outputs the sentiment label (POSITIVE/NEGATIVE) and confidence score.

"END"
