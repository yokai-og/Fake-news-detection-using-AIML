# Fake-news-detection-using-AIML
Fake News Detection using AI applies machine learning and LSTM deep learning models to classify news as real or fake. Using the Kaggle dataset with text preprocessing and TF-IDF features, the project compares multiple classifiers and shows LSTM achieves the highest accuracy.
<br>
Fake News Detection using Artificial Intelligence
📌 Overview

This project focuses on detecting fake news using Artificial Intelligence techniques. It applies both traditional machine learning models and deep learning (LSTM) to classify news articles as real or fake based on textual content.

🚀 Features

Text preprocessing (cleaning, tokenization, stopword removal)

TF-IDF feature extraction

Multiple ML models:

Logistic Regression

Naive Bayes

Random Forest

Deep Learning model:

Long Short-Term Memory (LSTM)

Performance comparison using accuracy, precision, recall, F1-score

Visualization using confusion matrices and accuracy plots

📂 Dataset

Source: Kaggle Fake News Dataset

Size: ~20,000 labeled news articles

Labels:

0 → Real

1 → Fake

🛠️ Technologies Used

Python 3

scikit-learn

TensorFlow / Keras

NLTK

pandas, numpy

matplotlib, seaborn

Google Colab (for training)

⚙️ Methodology

Load and preprocess text data

Extract features using TF-IDF

Train ML models and LSTM network

Evaluate models using standard metrics

Visualize results for comparison

📊 Results
Model	Accuracy
Logistic Regression	~86%
Naive Bayes	~83%
Random Forest	~89%
LSTM	~93%

The LSTM model performs best by capturing contextual and sequential patterns in text.

📌 Conclusion

This project demonstrates that deep learning models, especially LSTM, are highly effective for fake news detection. The framework is scalable and can be extended with transformer-based models and real-time deployment.

🔮 Future Enhancements

Integration of BERT / transformer models

Real-time news verification system

Explainable AI (LIME / SHAP)

Multilingual support

📄 License

This project is for academic and educational purposes.
