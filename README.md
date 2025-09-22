# 📊 Financial Document Classification  

This repository contains code and notebooks for **classifying financial documents** into categories such as Balance Sheets, Cash Flow, Income Statements, Notes, and Others using **NLP + Deep Learning (BiLSTM)**.  

The project demonstrates how to:  
- Extract and preprocess text from financial HTML documents  
- Generate word embeddings using **Word2Vec**  
- Handle imbalanced datasets using **SMOTETomek**  
- Train a **Bidirectional LSTM model** with TensorFlow/Keras  
- Evaluate model accuracy and visualize results  
- Make predictions on new financial documents  

---


---

## ⚙️ Features  
✅ Data extraction from HTML financial reports  
✅ Text preprocessing (tokenization, lemmatization, stopword removal)  
✅ Word2Vec embeddings for semantic representation  
✅ Handling imbalanced datasets with SMOTETomek  
✅ Bidirectional LSTM deep learning model  
✅ Accuracy/Loss visualization plots  
✅ Inference function for predicting new documents  

---

## 🚀 Getting Started  

### Prerequisites  
- Python 3.8+  
- Jupyter Notebook  

### Install Dependencies  
You can install all required libraries using:  
```bash
pip install -r requirements.txt

**### Running the Notebook**
git clone https://github.com/your-username/financial-document-classification.git
cd financial-document-classification
jupyter notebook


Open Financial_Document_Classification.ipynb and run the cells.

📊 Workflow

Data Collection – Download dataset from Kaggle

Preprocessing – Clean & tokenize text, remove stopwords

Embedding – Train Word2Vec model for vector representation

Balancing – Apply SMOTETomek to handle class imbalance

Model Training – Train BiLSTM on embeddings

Evaluation – Check accuracy, loss, and confusion matrix

Inference – Predict document type for unseen financial HTML files

📈 Results

High classification accuracy achieved using BiLSTM.

Visualization of training/validation accuracy and loss included.

📌 Future Improvements

Use Transformer models like BERT/RoBERTa

Experiment with larger and real-world financial datasets

Deploy as an API for real-time classification

🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss improvements.

