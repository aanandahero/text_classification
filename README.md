# 📚 Day 51 - Text Classification with Scikit-learn

This project demonstrates a basic text classification pipeline using the **20 Newsgroups dataset**, a well-known dataset for experimenting with text processing and Natural Language Processing (NLP) techniques.

We use **TF-IDF vectorization** to convert raw text into numerical features, and train a **Multinomial Naive Bayes** model to classify documents into one of 20 categories.

---

## 🚀 How to Run

1. **Install dependencies** (if not already installed):
   
   pip install scikit-learn
   
2. **Run the script**

  python text_classification.py


**What the Script Does**
Loads the 20 Newsgroups dataset using Scikit-learn.

Splits the data into training and testing sets.

Builds a machine learning pipeline using:

TfidfVectorizer to extract features from text.

MultinomialNB for classification.

Trains the model and evaluates accuracy.

Prints a classification report with performance metrics.


**Output Explanation**
Accuracy: 0.86
This means 86% of the test data was classified correctly.

**Classification Report**
Precision: How many selected items were correct?

Recall: How many correct items were selected?

F1-Score: Balance between precision and recall.

Support: Number of test samples in that class.

**Dataset Used**
We use the built-in 20 Newsgroups dataset from Scikit-learn:

Contains 18,000+ newsgroup articles.

Spread across 20 categories like sci.space, rec.sport.hockey, comp.graphics, etc.

No need to download anything — it loads automatically.

**Tools Used**
Python

Scikit-learn

TfidfVectorizer

Multinomial Naive Bayes




  
