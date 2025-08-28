# Movie Review Sentiment Analysis

A comprehensive machine learning project for analyzing sentiment in movie reviews using multiple classification algorithms and advanced text preprocessing techniques.

## Overview

This project implements sentiment analysis on movie reviews using the NLTK movie reviews corpus. It compares the performance of multiple machine learning models and provides detailed visualizations of the results, including word clouds, confusion matrices, and ROC curves.

## Features

- **Advanced Text Preprocessing**: Lemmatization, stopword removal, and POS tagging
- **Multiple ML Models**: Naive Bayes, Logistic Regression, Linear SVM, and Random Forest
- **Comprehensive Evaluation**: Accuracy scores, classification reports, confusion matrices, and ROC curves
- **Data Visualization**: Model comparison charts, word clouds for positive/negative reviews
- **Interactive Testing**: Test custom movie reviews with the trained model

## Dependencies

```python
nltk
pandas
scikit-learn
matplotlib
seaborn
wordcloud
numpy
```

## Installation

1. Clone this repository or download the notebook
2. Install required packages:
```bash
pip install nltk pandas scikit-learn matplotlib seaborn wordcloud numpy
```

3. Run the notebook - NLTK datasets will be automatically downloaded:
   - movie_reviews
   - punkt
   - stopwords
   - wordnet
   - omw-1.4
   - punkt_tab
   - averaged_perceptron_tagger_eng

## Usage

### Running the Analysis

1. **Data Preparation**: The script automatically loads and preprocesses the NLTK movie reviews dataset
2. **Text Cleaning**: Advanced preprocessing including lemmatization and POS tagging
3. **Feature Extraction**: TF-IDF vectorization with optimized parameters
4. **Model Training**: Trains and compares multiple classification models
5. **Evaluation**: Generates comprehensive performance metrics and visualizations

### Testing Custom Reviews

The project includes a testing section where you can analyze custom movie reviews:

```python
test_review = ["Your custom movie review here"]
# The model will predict sentiment as Positive or Negative
```

## Models Implemented

| Model | Algorithm | Purpose |
|-------|-----------|---------|
| **Naive Bayes** | MultinomialNB | Baseline probabilistic classifier |
| **Logistic Regression** | LogisticRegression | Linear classification with regularization |
| **Linear SVM** | LinearSVC | Support Vector Machine for text classification |
| **Random Forest** | RandomForestClassifier | Ensemble method for robust predictions |

## Text Preprocessing Pipeline

1. **Lowercasing**: Convert all text to lowercase
2. **Special Character Removal**: Remove numbers and special characters
3. **Tokenization**: Split text into individual words
4. **POS Tagging**: Identify parts of speech for better lemmatization
5. **Lemmatization**: Reduce words to their root forms
6. **Stopword Removal**: Remove common English stopwords
7. **Length Filtering**: Keep only words longer than 2 characters

## Evaluation Metrics

- **Accuracy Score**: Overall classification accuracy
- **Classification Report**: Precision, recall, and F1-score for each class
- **Confusion Matrix**: Visual representation of correct/incorrect predictions
- **ROC Curve**: Receiver Operating Characteristic curve with AUC score
- **Model Comparison**: Side-by-side accuracy comparison of all models

## Visualizations

The project generates several insightful visualizations:

1. **Model Accuracy Comparison**: Bar chart comparing all models
2. **Word Clouds**: Separate clouds for positive and negative reviews
3. **ROC Curve**: Performance curve for the Naive Bayes model
4. **Confusion Matrix**: Classification accuracy breakdown

## Results

The project typically achieves:
- **Naive Bayes**: ~85% accuracy (baseline)
- **Logistic Regression**: ~87% accuracy
- **Linear SVM**: ~86% accuracy  
- **Random Forest**: ~85% accuracy

## Key Features of Implementation

- **Balanced Dataset**: Equal distribution of positive and negative reviews
- **Optimized TF-IDF**: Uses unigrams and bigrams with frequency filtering
- **Cross-Model Comparison**: Systematic evaluation of multiple approaches
- **Comprehensive Metrics**: Multiple evaluation perspectives
- **Visual Analysis**: Word frequency and model performance visualization

## Dataset Information

- **Source**: NLTK Movie Reviews Corpus
- **Size**: 2000 movie reviews (1000 positive, 1000 negative)
- **Split**: 80% training, 20% testing
- **Preprocessing**: Advanced NLP techniques applied

## Contributing

Feel free to fork this project and submit pull requests for improvements such as:
- Additional machine learning models
- Enhanced text preprocessing techniques
- New visualization methods
- Performance optimizations

## License

This project is open source and available under standard academic use.

## References

- NLTK Movie Reviews Corpus
- Scikit-learn Documentation
- Natural Language Processing best practices 


---

**Note**: This project is designed for educational purposes and demonstrates practical implementation of sentiment analysis techniques using traditional machine learning approaches.
