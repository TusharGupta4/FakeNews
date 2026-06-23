# 🔍 Fake News Detection System

A deep learning-based fake news detection application using a **Dual-Branch LSTM Neural Network** trained on the WELFake dataset. This project achieves **95.16% accuracy** and **0.9887 ROC-AUC score** in classifying news articles as real or fake.

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 95.16% |
| **ROC-AUC Score** | 0.9887 |
| **Precision (Real News)** | 0.95 |
| **Precision (Fake News)** | 0.96 |
| **Recall (Real News)** | 0.96 |
| **Recall (Fake News)** | 0.95 |
| **F1-Score** | 0.95 |

### Confusion Matrix
```
                    Predicted
                Real     Fake
Actual  Real  [ 11201     492 ]
        Fake  [   650   11265 ]
```

## 🏗️ Model Architecture

The model uses a **Dual-Branch LSTM** architecture to process news titles and article text separately:

```
Title Input (20 tokens)          Article Text Input (200 tokens)
        ↓                                      ↓
   Embedding Layer                    Embedding Layer
   (64 dimensions)                    (64 dimensions)
        ↓                                      ↓
   LSTM Layer                         LSTM Layer
   (64 units)                         (64 units)
        ↓                                      ↓
   Dropout (0.4)                      Dropout (0.4)
        ↓                                      ↓
        └──────────→ Concatenate ←───────────┘
                        ↓
                  Dense Layer (64 units, ReLU)
                        ↓
                  Dropout (0.3)
                        ↓
              Output Layer (Sigmoid)
                        ↓
            Classification (0 = Real, 1 = Fake)
```

### Key Architecture Features:
- **Separate branches** for title and text processing
- **Embedding dimension**: 64
- **LSTM units**: 64 per branch
- **Dropout rates**: 0.4 (LSTM), 0.3 (final)
- **Vocabulary size**: 15,000 most common words
- **Title max length**: 20 tokens
- **Text max length**: 200 tokens

## 📂 Dataset

**WELFake Dataset**
- **Total articles**: 71,537
- **Training samples**: 47,930 (67%)
- **Testing samples**: 23,607 (33%)
- **Classes**: 2 (Real News: 0, Fake News: 1)
- **Features**: Title, Text, Label

## ✨ Features

- ✅ Real-time fake news detection with confidence scores
- ✅ Dual-input model (Title + Article text)
- ✅ Three-tier classification (Real, Fake, Uncertain)
- ✅ Interactive Streamlit web interface
- ✅ Text preprocessing with lemmatization
- ✅ Customizable decision threshold (default: 0.70)
- ✅ Uncertainty handling for borderline cases

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/fake-news-detection.git
cd FakeNews
```

### Step 2: Create Virtual Environment
```bash
python -m venv .venv

# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data
```python
python -c "
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
"
```

## 🚀 Usage

### Run the Streamlit App
```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`

### Using the Web Interface

1. **Input News Article**: 
   - Paste the article title in the first field
   - Paste the full article text in the second field

2. **Get Prediction**:
   - Click "Analyze News" button
   - View result: 🟢 Real News | 🔴 Fake News | 🟡 Uncertain

3. **Interpretation**:
   - **Confidence Score**: Probability strength (0.0 - 1.0)
   - **Probability**: Raw model output probability

### Threshold Logic
- **Prob > 0.70** → FAKE (high confidence)
- **Prob < 0.40** → REAL (high confidence)
- **0.40 ≤ Prob ≤ 0.70** → UNCERTAIN (review manually)

## 📝 Text Preprocessing

The model uses comprehensive text preprocessing:

1. **Cleaning**: Remove special characters, keep only alphabets
2. **Lowercase**: Convert to lowercase for consistency
3. **Lemmatization**: Reduce words to base form (e.g., "running" → "run")
4. **Stop Words**: Remove common words except negations:
   - Kept: "not", "no", "nor", "never", "n't"
   - Removed: "the", "a", "is", "and", etc.

## 🔄 Training Pipeline

### Step 1: Data Loading & Cleaning
```bash
python fake_news.py
```

The training script performs:
- Load WELFake dataset
- Remove null values
- Verify labels (0 or 1 only)
- Split into 67% train, 33% test

### Step 2: Model Training
- Tokenization of titles and text separately
- Padding sequences to fixed lengths
- Train Dual-Branch LSTM model
- Use EarlyStopping to prevent overfitting
- Save model as `fake_news_model.h5`
- Save tokenizer as `tokenizer.pkl`

## 📦 Project Structure

```
FakeNews/
├── app.py                      # Streamlit web application
├── fake_news.py               # Training script
├── fake_news_model.h5         # Trained model (binary format)
├── tokenizer.pkl              # Tokenizer for text encoding
├── WELFake_Dataset.csv        # Training dataset
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── .gitignore                 # Git ignore rules
```

## 📋 Requirements

```
tensorflow>=2.11.0
keras>=2.11.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
nltk>=3.8.1
streamlit>=1.20.0
```

## 🎯 Performance Analysis

### Strengths:
- ✅ Excellent precision on both classes (95-96%)
- ✅ Balanced recall (95-96%)
- ✅ High ROC-AUC indicates strong discrimination ability
- ✅ Model handles both real and fake news equally well

### Considerations:
- 📌 ~4.2% false positives (real news classified as fake)
- 📌 ~5% false negatives (fake news classified as real)
- 📌 Works best with English text articles

## 🔐 Model Limitations

1. **Language**: Currently optimized for English text only
2. **Article Format**: Best results with standard news article format
3. **Updates**: Model trained on historical data; may need retraining for emerging fake news patterns
4. **Satire**: May confuse satirical articles with fake news
5. **Context**: Relies on text content, not external verification


## 📚 References

1. **Dataset**: WELFake - A Real-world Annotated Dataset for Fake News Detection
2. **Architecture**: Dual-Branch LSTM for sequence classification
3. **Framework**: TensorFlow/Keras, Streamlit




