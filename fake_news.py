# ══════════════════════════════════════════════════════════════
# FAKE NEWS DETECTION — DUAL BRANCH MODEL
# Title + Text alag alag process honge
# ══════════════════════════════════════════════════════════════

# ── 1. IMPORTS ───────────────────────────────────────────────
import pandas as pd
import numpy as np
import re
import pickle
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import tensorflow as tf
from tensorflow.keras.layers import (Input, Embedding, LSTM,Dense, Concatenate, Dropout)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix,classification_report, roc_auc_score)

# ── 2. LOAD DATA ─────────────────────────────────────────────
df = pd.read_csv('WELFake_Dataset.csv')
print("Shape:", df.shape)
print(df.head())

# ── 3. CLEAN DATA ────────────────────────────────────────────
df = df.dropna()

# Also drop rows where label is not 0 or 1
df = df[df['label'].isin([0, 1])]
print("After cleaning:", df.shape)

# ── 4. SEPARATE TITLE AND TEXT ───────────────────────────────
# KEY DIFFERENCE from before — NOT combining them
title_raw = df['title'].astype(str)
text_raw  = df['text'].astype(str)
y         = df['label'].values

# ── 5. PREPROCESSING ─────────────────────────────────────────
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Keep negation words
stop_words.discard('not')
stop_words.discard('no')
stop_words.discard('nor')
stop_words.discard('never')
stop_words.discard("n't")

def preprocess(text):
    review = re.sub('[^a-zA-Z]', ' ', str(text))
    review = review.lower().split()
    review = [lemmatizer.lemmatize(w) for w in review if w not in stop_words]
    return ' '.join(review)

print("\nPreprocessing titles...")
title_corpus = []
for i, title in enumerate(title_raw):
    title_corpus.append(preprocess(title))
    if i % 10000 == 0:
        print(f"  Titles: {i}/{len(title_raw)}")

print("\nPreprocessing article bodies...")
text_corpus = []
for i, text in enumerate(text_raw):
    text_corpus.append(preprocess(text))
    if i % 10000 == 0:
        print(f"  Texts: {i}/{len(text_raw)}")

print("Preprocessing done!")

# ── 6. TOKENIZER ─────────────────────────────────────────────
# Single tokenizer for both title and text
# So same word gets same index in both
voc_size = 15000

tokenizer = Tokenizer(num_words=voc_size)
tokenizer.fit_on_texts(title_corpus + text_corpus)  # fit on both
print(f"Vocabulary size: {len(tokenizer.word_index)}")

# ── 7. SEQUENCES ─────────────────────────────────────────────
# Title — short maxlen (titles are short)
title_maxlen = 20

title_seqs   = tokenizer.texts_to_sequences(title_corpus)
title_padded = pad_sequences(title_seqs,maxlen=title_maxlen,padding='post',truncating='post')

# Text — longer maxlen (articles are long)
text_maxlen  = 200

text_seqs    = tokenizer.texts_to_sequences(text_corpus)
text_padded  = pad_sequences(text_seqs,maxlen=text_maxlen,padding='post',truncating='pre')

print(f"Title padded shape: {title_padded.shape}")
print(f"Text padded shape:  {text_padded.shape}")

# ── 8. TRAIN TEST SPLIT ──────────────────────────────────────
# Split both title and text together using same indices
from sklearn.model_selection import train_test_split

indices = np.arange(len(y))

(idx_train, idx_test) = train_test_split(indices,test_size=0.33,random_state=42)

title_train = title_padded[idx_train]
title_test  = title_padded[idx_test]
text_train  = text_padded[idx_train]
text_test   = text_padded[idx_test]
y_train     = y[idx_train]
y_test      = y[idx_test]

print(f"\nTraining samples: {len(y_train)}")
print(f"Testing samples:  {len(y_test)}")

# ── 9. BUILD DUAL BRANCH MODEL ───────────────────────────────
embedding_dim = 64

# ── Branch 1 — Title ─────────────────────────────────────────
title_input = Input(shape=(title_maxlen,), name='title_input')
title_emb   = Embedding(voc_size,embedding_dim,input_length=title_maxlen)(title_input)
title_lstm  = LSTM(64, return_sequences=False)(title_emb)
title_drop  = Dropout(0.4)(title_lstm)

# ── Branch 2 — Text ──────────────────────────────────────────
text_input  = Input(shape=(text_maxlen,), name='text_input')
text_emb    = Embedding(voc_size,embedding_dim,input_length=text_maxlen)(text_input)
text_lstm   = LSTM(64, return_sequences=False)(text_emb)
text_drop   = Dropout(0.4)(text_lstm)

# ── Merge Both Branches ───────────────────────────────────────
# Concatenate: [64 numbers from title] + [64 numbers from text]
# = 128 numbers total
merged     = Concatenate()([title_drop, text_drop])
dense1     = Dense(64, activation='relu')(merged)
drop_final = Dropout(0.3)(dense1)
output     = Dense(1, activation='sigmoid')(drop_final)

# ── Build Model ───────────────────────────────────────────────
model = Model(
    inputs=[title_input, text_input],
    outputs=output
)

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print(model.summary())

# ── 10. EARLY STOPPING ───────────────────────────────────────
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=2,
    restore_best_weights=True
)

# ── 11. TRAIN MODEL ──────────────────────────────────────────
history = model.fit(
    [title_train, text_train],   # both inputs
    y_train,
    validation_data=([title_test, text_test], y_test),
    epochs=10,
    batch_size=64,
    callbacks=[early_stop]
)

# ── 12. EVALUATE ─────────────────────────────────────────────
y_pred_prob = model.predict([title_test, text_test])
y_pred      = np.where(y_pred_prob > 0.5, 1, 0)

print("\n" + "="*55)
print("MODEL EVALUATION")
print("="*55)
print(f"Accuracy:      {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_prob):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred,target_names=['Real News', 'Fake News']))

# ── 13. SAVE ─────────────────────────────────────────────────
model.save('fake_news_model.h5')
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("\nModel saved: fake_news_model.h5")
print("Tokenizer saved: tokenizer.pkl")

