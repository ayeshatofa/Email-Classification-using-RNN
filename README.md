# 📧 Email Classification Using RNN & LSTM
### Neural Networks & Fuzzy Logic (NNFL) Project
This project implements an end-to-end email classification system using Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) networks. The system classifies emails as ham (legitimate) or spam using deep learning.

The project includes:
- Dataset loading & cleaning
- Exploratory Data Analysis (EDA)
- Text preprocessing
- RNN and LSTM model training
- Evaluation (accuracy, confusion matrix, precision, recall, F1)
- Custom email prediction pipeline
## 🚀 Features
- Raw email text preprocessing (tokenization, stemming, stopword removal, n-grams)
- Train/validation/test split with stratification
- Deep learning-based text classification
- RNN baseline model
- LSTM enhanced model
- Model checkpointing + early stopping
- Confusion matrix & metrics visualization
- Custom email prediction function
- Ready for deployment/inference
## 🧠 Project Architecture
```
Email_Classification_NNFL/
│
├── data/
│   └── Emailscam.csv
│
├── notebooks/
│   └── email_classification.ipynb
│
├── models/
│   ├── best_rnn.weights.h5
│   └── best_lstm.weights.h5
│
├── results/
│   ├── confusion_matrix.png
│   ├── training_curves.png
│   └── classification_report.txt
│
├── README.md
└── requirements.txt
```
## 📊 Dataset
- Source: Mendeley Data
- Classes:
  - ham: legitimate messages
  - spam: unsolicited / fraudulent messages
    
Load dataset:
```
df = pd.read_csv(
    "https://raw.githubusercontent.com/ayeshatofa/Email-Classification-using-RNN/main/Emailscam.csv",
    encoding="latin1"
)
```
## 🔍 Exploratory Data Analysis (EDA)
The notebook includes visualizations for:
- Class distribution
- Word count analysis
- Punctuation frequency
- Stopword count
- URL frequency
- Frequent spam vs ham words

## 🛠 Text Preprocessing
Preprocessing steps include:
- Lowercasing
- Emoji removal
- Punctuation removal
- Stopword removal
- Stemming (PorterStemmer)
- Bigram generation
- Tokenization
- Sequence padding

```
processed_sentence = preprocessing(text)
processed_sentence = stopwordRemoval(processed_sentence)
processed_sentence = stem_text(processed_sentence)
processed_sentence = ' '.join(generate_ngrams(processed_sentence, n=2))
```

## 🤖 Model Development
### SimpleRNN Model
- Embedding layer
- SimpleRNN(64)
- Dense layers with dropout
### LSTM Model
- Embedding layer
- LSTM(64) with dropout
- Dense classifier
### Training Setup
- Loss: binary_crossentropy
- Optimizer: adam
- Batch size: 32
- Epochs: 20
- EarlyStopping
- ModelCheckpoint

## 📈 Evaluation
### RNN Performance

| Metric               | Score   |
|----------------------|---------|
| **Accuracy**         | 0.9731  |
| **Loss**             | 0.0949  |
| **Precision (weighted)** | 0.7492  |
| **Recall (weighted)**    | 0.8656  |
| **F1 Score**             | 0.8032  |

### LSTM Performance
| Metric               | Score   |
|----------------------|---------|
| **Accuracy**         | 0.9821  |
| **Loss**             | 0.0705  |
| **Precision (weighted)** | 0.7492  |
| **Recall (weighted)**    | 0.8656  |
| **F1 Score**             | 0.8032  |
### Confusion Matrix (Both Models)
[[483   0]
 [ 75   0]]

**⚠ Note**: Both models predicted all instances as ham due to dataset imbalance and heavy preprocessing, causing 0% recall on spam.

## 🧪 Custom Email Prediction
```
pred = model.predict(padded_seq)
pred_class = label_encoder.classes_[int(pred[0][0] > 0.5)]
print("Predicted Label:", pred_class)
```

## ⚠ Limitations
- Dataset imbalance → poor spam detection
- Heavy preprocessing removed useful patterns
- No class weighting used
- Simple RNN/LSTM architecture; lacks attention/transformers

## 🔮 Future Improvements
- Use class weights (e.g., class_weight='balanced')
- Apply SMOTE or oversampling for spam
- Include metadata features (sender, domain)
- Switch to BERT, DistilBERT, or Transformer models
- Refine preprocessing (keep URLs, moderate stopword removal)

## ▶ How to Run
1.	Clone the repo:
```
git clone https://github.com/ayeshatofa/Email-Classification-using-RNN.git
cd Email-Classification-using-RNN
```
2.	Install dependencies:
```
pip install -r requirements.txt
```
3.	Open the notebook:
```
code CODE_2257_1002_1026.ipynb
```
4.	Run all cells to train models / test predictions.
