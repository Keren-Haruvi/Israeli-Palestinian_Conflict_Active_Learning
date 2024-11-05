import os
import ast
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import DebertaTokenizer, DebertaModel, BertTokenizer, BertModel,  RobertaTokenizer, RobertaModel, AutoModelForSequenceClassification, AutoTokenizer, pipeline

import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)


def get_tfidf_embedding(text, vectorizer):
    """
    transforms the input text into a TF-IDF representation
    """
    tfidf_embedding = vectorizer.transform([text])
    return tfidf_embedding.toarray().squeeze().tolist()


def get_bert_embedding(text):
    """
    tokenizes the input text and passes it through a pre-trained BERT model 
    """    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model.eval()
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings.tolist()


def get_deberta_embedding(text):
    """
    tokenizes the input text and passes it through a pre-trained deBERTa model 
    """    
    tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
    Deberta_model = DebertaModel.from_pretrained('microsoft/deberta-base')
    Deberta_model.eval()
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = Deberta_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings.tolist()


def get_roberta_embedding(text):
    """
    tokenizes the input text and passes it through a pre-trained roBERTa model 
    """    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')
    model.eval()
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings.tolist()


def embedding_pipeline(df, text_col):
    """
    generate embeddings for a specified text column using various embedding models
    """
    texts = df[text_col].tolist()

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(texts)
    df['tfidf'] = df[text_col].apply(lambda text: get_tfidf_embedding(text, tfidf_vectorizer))
    df['bert'] = df[text_col].apply(lambda text: get_bert_embedding(text))
    df['deberta'] = df[text_col].apply(lambda text: get_deberta_embedding(text))
    df['roberta'] = df[text_col].apply(lambda text: get_roberta_embedding(text))

    return df


def preprocess_text_for_sentiment_analysis(text, tokenizer, max_length=512):
    """
    preprocess text for sentiment analysis by tokenizing and truncating to a specified maximum length
    """
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    return tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=True)
    

def sentiment_analysis_certainty(df, text_col):
    """
    perform sentiment analysis on a specified text column and calculate the certainty score for each sentiment prediction.
    """
    model_name = "finiteautomata/bertweet-base-sentiment-analysis"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, truncation=True, device='cuda')

    texts = df[text_col].apply(lambda x: preprocess_text_for_sentiment_analysis(x, tokenizer)).to_list()
    results = sentiment_analysis(texts)

    sentiment_certainty_list = []
    for text, result in zip(texts, results):
        sentiment_certainty_list.append(result['score'])
        
    df['sentiment_certainty'] = sentiment_certainty_list
    return df


def create_process_DataFrame(path, test_size=50):
    # df contains records extracted from podcast transcriptions
    nlp_df_path = os.path.join(path, 'nlp_df.csv')
    df1 = pd.read_csv(nlp_df_path)
    
    df1['label'] = df1['label'].apply(ast.literal_eval)
    df1 = df1[df1['label'].apply((lambda x: (x[0] == 1 or x[1] == 1) and not (x[0] == 1 and x[1] == 1)))]
    df1['y'] = df1['label'].apply(lambda x: 1 if x[0]==1 else 0)
    df1 = df1[['text', 'y']]

    # df contains records extracted from reddit
    reddit_df_path = os.path.join(path, 'reddit_df.csv')
    df2 = pd.read_csv(reddit_df_path)
    
    df2 = df2[df2['label'] != 'not relevant']
    df2['y'] = df2['label'].apply(lambda x: 1 if x=='israel' else 0)
    df2['text'] = df2['comment']
    df2 = df2[['text', 'y']]

    # df contains records generated using CoT and GPT
    gpt_df_path = os.path.join(path, 'GPT_df.csv')
    df3 = pd.read_csv(gpt_df_path)

    df = pd.concat([df1, df2, df3], axis=0).reset_index(drop=True)

    # compute sentiment analysis certainties
    df = sentiment_analysis_certainty(df, 'text')

    # create different embeddings
    df = embedding_pipeline(df, 'text')
    
    # calculate num of words in every sentance
    df['num_of_words'] = df['text'].apply(lambda x: len(x.split(' ')))

    # define test set
    df['group'] = 'pool'
    test_indices = np.random.choice(df.index, size=test_size, replace=False)
    df.loc[test_indices, 'group'] = 'test'

    return df


def create_data_analysis(df):
    """
    creates visualizations for data analysis
    """
    # num of words distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df['num_of_words'], kde=True)
    plt.title('Histogram of Number of Words')
    plt.xlabel('Number of Words')
    plt.ylabel('Count')
    plt.show()

    # label distribution
    plt.figure(figsize=(8, 5))
    sns.countplot( data=df, x='y')
    plt.title('Count Plot of Labels')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.xticks(ticks=[0, 1], labels=['Pro Palestine', 'Pro Israel'])
    plt.show()


if __name__ == "__main__":
    current_directory = os.getcwd()
    datasets_path = os.path.join(current_directory, 'Datasets')
    df = create_process_DataFrame(datasets_path)
    
    relative_path = os.path.join(current_directory, 'Datasets', 'processed_df.csv')
    df.to_csv(relative_path, index=False)
    # df.to_csv('/home/student/Project/Datasets/processed_df.csv', index=False)