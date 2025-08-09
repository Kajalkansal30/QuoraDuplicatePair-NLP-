# Quora Question Pairs - Duplicate Detection Project

This project focuses on building a machine learning model to identify duplicate questions on Quora using Natural Language Processing (NLP) techniques. The goal is to predict whether two given questions are semantically identical.

## 🎯 Project Overview

The Quora Question Pairs dataset contains pairs of questions labeled as either:
- **1**: Duplicate questions (semantically identical)
- **0**: Non-duplicate questions (different meaning)

## 📁 Project Structure

```
QuoraProject/
├── train.csv                    # Training dataset with question pairs
├── initial_EDA.ipynb           # Exploratory Data Analysis
├── only-bow.ipynb              # Basic Bag-of-Words model
├── bow-with-basic-features.ipynb # BoW with engineered features
├── bow-with-preprocessing-and-advanced-features.ipynb # Advanced preprocessing model
├── model.pkl                   # Trained model file
├── cv.pkl                      # CountVectorizer object
├── Untitled.ipynb              # Additional experiments
└── README.md                   # This file
```

## 📊 Dataset Description

**File**: `train.csv`
- **Size**: ~404,290 question pairs
- **Columns**:
  - `id`: Unique identifier for each question pair
  - `qid1`: Question ID for the first question
  - `qid2`: Question ID for the second question
  - `question1`: First question text
  - `question2`: Second question text
  - `is_duplicate`: Target variable (1 for duplicate, 0 for non-duplicate)

## 🔍 Exploratory Data Analysis

Key findings from the initial EDA:
- **Class Imbalance**: ~63% non-duplicate vs ~37% duplicate questions
- **Missing Values**: Some questions have missing text
- **Question Repetition**: Many questions appear multiple times across different pairs
- **Text Length**: Questions vary significantly in length and complexity

## 🚀 Model Development Pipeline

### 1. Basic Bag-of-Words Model (`only-bow.ipynb`)
- **Approach**: Simple text vectorization using CountVectorizer
- **Features**: 3000 most frequent words
- **Models**: RandomForest and XGBoost
- **Accuracy**: ~75-78%

### 2. Advanced Feature Engineering (`bow-with-preprocessing-and-advanced-features.ipynb`)
- **Text Preprocessing**:
  - Lowercase conversion
  - Contraction expansion (e.g., "can't" → "can not")
  - Special character handling
  - HTML tag removal
  - Number normalization

- **Engineered Features** (22 total):
  - **Basic Features**:
    - Question length (character count)
    - Word count
    - Common words between questions
    - Word share ratio
  
  - **Token Features**:
    - Common word count (non-stopwords)
    - Common stopword count
    - Common token count
    - First/last word matching
  
  - **Length Features**:
    - Absolute length difference
    - Average token length
    - Longest common substring ratio
  
  - **Fuzzy Features**:
    - Fuzzy ratio
    - Partial ratio
    - Token sort ratio
    - Token set ratio

- **Dimensionality Reduction**: t-SNE visualization for feature analysis
- **Models**: RandomForest and XGBoost
- **Accuracy**: ~82-85%

## 🧪 Model Performance

| Model | Features | Accuracy |
|-------|----------|----------|
| RandomForest | Basic BoW | ~75% |
| XGBoost | Basic BoW | ~78% |
| RandomForest | Advanced Features + BoW | ~82% |
| XGBoost | Advanced Features + BoW | ~85% |

## 🔧 Usage

### Loading the Trained Model

```python
import pickle

# Load the trained model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open('cv.pkl', 'rb'))

# Make predictions on new question pairs
q1 = "What is the capital of India?"
q2 = "Which city is the capital of India?"

# The query_point_creator function handles all preprocessing and feature engineering
prediction = model.predict(query_point_creator(q1, q2))
```

### Prediction Function

The `query_point_creator` function automatically:
1. Preprocesses both questions
2. Extracts all 22 engineered features
3. Applies the CountVectorizer
4. Returns the complete feature vector for prediction

## 📈 Key Insights

1. **Feature Engineering Impact**: Advanced features improved accuracy by ~7-10%
2. **Class Imbalance**: Models handle the 63-37 imbalance reasonably well
3. **Text Similarity**: Fuzzy matching features are particularly effective
4. **Preprocessing**: Text cleaning significantly improves model performance

## 🛠️ Dependencies

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
nltk
beautifulsoup4
fuzzywuzzy
distance
```

## 🚀 Future Improvements

- **Deep Learning**: Implement Siamese networks or BERT-based models
- **Ensemble Methods**: Combine multiple models for better performance
- **Hyperparameter Tuning**: Grid search for optimal parameters
- **Data Augmentation**: Generate synthetic duplicate pairs
- **Cross-validation**: Implement k-fold validation for robust evaluation

## 🤝 Contributing

Feel free to contribute by:
- Adding new features or preprocessing techniques
- Implementing different model architectures
- Improving evaluation metrics
- Adding comprehensive unit tests

## 📄 License

This project is open source and available under the MIT License.
