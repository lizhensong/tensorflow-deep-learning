import os
import random
import numpy as np
import matplotlib.pyplot as plt


def load_imdb_sentiment_analysis_dataset(data_path, seed=123):
    imdb_data_path = os.path.join(data_path, 'aclImdb')
    texts = []
    labels = []
    for category in ['pos', 'neg']:
        train_path = os.path.join(imdb_data_path, 'train', category)
        for fname in sorted(os.listdir(train_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(train_path, fname), encoding='utf-8') as f:
                    texts.append(f.read())
                labels.append(0 if category == 'neg' else 1)
    for category in ['pos', 'neg']:
        test_path = os.path.join(imdb_data_path, 'test', category)
        for fname in sorted(os.listdir(test_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(test_path, fname), encoding='utf-8') as f:
                    texts.append(f.read())
                labels.append(0 if category == 'neg' else 1)
    random.seed(seed)
    random.shuffle(texts)
    random.seed(seed)
    random.shuffle(labels)

    return ((texts[:int(len(texts) * 0.6)], np.array(labels)[:int(len(texts) * 0.6)]),
            (texts[int(len(texts) * 0.6):int(len(texts) * 0.8)],
             np.array(labels)[int(len(texts) * 0.6):int(len(texts) * 0.8)]),
            (texts[int(len(texts) * 0.8):], np.array(labels)[int(len(texts) * 0.8):]))


def get_num_words_per_sample(sample_texts):
    num_words = [len(s.split()) for s in sample_texts]
    plt.hist(num_words, 50)
    plt.xlabel('words number of a sample')
    plt.ylabel('Number of samples')
    plt.title('Sample words number')
    plt.show()


def plot_sample_length_distribution(sample_texts):
    plt.hist([len(s) for s in sample_texts], 50)
    plt.xlabel('Length of a sample')
    plt.ylabel('Number of samples')
    plt.title('Sample length distribution')
    plt.show()


def get_s_w(sample_texts):
    word_num = 0
    for s in sample_texts:
        word_num += len(s.split())
    s_w = len(sample_texts) / (word_num / len(sample_texts))
    return s_w

# a = load_imdb_sentiment_analysis_dataset('D:\Python_Work_Space\learning-data\movie-imbd-Initially')
# print(get_s_w(a[0][0]))
# get_num_words_per_sample(a[0][0])
# plot_sample_length_distribution(a[0][0])
