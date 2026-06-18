from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# AG News with 4 classes
LABEL_NAMES = ['World', 'Sports', 'Business', 'Sci/Tech']


def load_ag_news():
    
    #load data from library

    from datasets import load_dataset
    dataset = load_dataset("fancyzhx/ag_news")

    train_texts = [item['text'] for item in dataset['train']]
    train_labels = [item['label'] for item in dataset['train']]

    test_texts = [item['text'] for item in dataset['test']]
    test_labels = [item['label'] for item in dataset['test']]

    return train_texts, train_labels, test_texts, test_labels