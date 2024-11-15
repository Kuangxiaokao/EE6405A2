#Sentiment Analysis with Transformer Models
This project implements sentiment analysis using models（CNN,RNN,LSTM,GRU,Bi-lstm，Transformer）  on Yelp, SST-2, and IMDB datasets. The goal is to classify text reviews as positive or negative, leveraging pre-trained language models for efficient and accurate text classification.

#Features
Preprocessing: Tokenization, text cleaning, and handling class imbalances.
Transformer Models: Fine-tuning pre-trained models such as BERT for sentiment classification.
Evaluation Metrics: Comprehensive metrics including accuracy, loss, recall, precision, and AUC.

#Dataset-Specific Analysis:
Addressing challenges in diverse datasets with varying review lengths and sentiment distributions.
#Datasets
Yelp Reviews: Detailed and verbose reviews.
SST-2 (Stanford Sentiment Treebank): Short, concise movie reviews.
IMDB Reviews: Mixed-sentiment and polarized reviews.
#Results
Metrics: Models achieved high accuracy and AUC across datasets, with balanced precision-recall performance.
Visualization: Loss and metric trends during training are saved in the plots/ directory.
#Future Work
Dataset augmentation and multi-task learning.
Exploring advanced transformer architectures (e.g., RoBERTa, GPT).
Incorporating explainability tools for attention visualization.
