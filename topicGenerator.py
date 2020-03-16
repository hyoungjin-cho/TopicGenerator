import os
import nltk
import re
import numpy as np
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer

class Documents:
    docs = None # list of document objects
    word2Idx = None # a map ; word : index
    idx2Word = None # a list of words ; index : word
    wordCount = None # word counts
    class Document:
        name = None # document name (full path to this file)
        docWords = None # bag of words representation (contains indices of words instead of words inself)
        def __init__(self, path, word2Idx, idx2Word, wordCount):
            self.name = path
            self.docWords = []
            try:
                with open(path, 'r') as f:
                    text = f.read()
            except UnicodeDecodeError:
                with open(path, 'r', encoding = "ISO-8859-1") as f:
                    text = f.read()
            # Preprocessing text data
            # converting to lower case, removing special characters, and single alphabet
            text = text.lower()
            text = re.sub(r'\W', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r"\b[a-zA-Z]\b", "", text)
            # tokenize
            words_tmp = nltk.tokenize.word_tokenize(text)
            # remove stopwords/numbers, and convert its baseform
            ps =PorterStemmer()
            words = [ps.stem(w) for w in words_tmp if not w.isdigit() and not w in ENGLISH_STOP_WORDS]
            # Word to Index; assign each words to unique integer across all documents
            for i in range(len(words)):
                if words[i] not in word2Idx:
                    idx = len(word2Idx)
                    word2Idx[words[i]] = idx
                    idx2Word.append(words[i])
                    self.docWords.append(idx)
                    wordCount[words[i]] = 1
                else:
                    self.docWords.append(word2Idx[words[i]])
                    wordCount[words[i]] += 1

    def __init__(self):
        self.docs = []
        self.word2Idx = {}
        self.idx2Word = []
        self.wordCount = {}

    def readDocs(self, path):
        files = os.listdir(path)
        for file in sorted(files):
            if not file.startswith('.'):
                doc = self.Document(path + file, self.word2Idx, self.idx2Word, self.wordCount)
                self.docs.append(doc)

class LDA:
    K = None # number of topics
    M = None # number of documents
    W = None # number of unique words in all docs
    z = None # MxN matrix; z[m][n] = a topic (0 to K-1), where N is a length of document m
    alpha = None # document-topic dirichlet pior parameter
    beta = None # topic-word dirichlet pior parameter
    MK = None # MxK matrix; MK[m][k] = total number of words that assigned to topic k in document m
    KW = None # KxW matrix; KW[k][w] = total counts of word w in topic k
    theta = None # MxK matrix; document-topic distribution parameters
    phi = None # KxW matrix; topic-word distribution parameters
    num_iter = None # number of iterations
    docs = None # documents object

    def __init__(self, alpha=1, beta=1, num_iter=1000, K=10, documents=None):
        self.K = K
        self.M = len(documents.docs)
        self.W = len(documents.word2Idx)
        self.z = [[] for m in range(self.M)]
        self.alpha = alpha
        self.beta = beta
        self.MK = np.zeros((self.M,self.K))
        self.KW = np.zeros((self.K,self.W))
        self.theta = np.zeros((self.M,self.K))
        self.phi = np.zeros((self.K,self.W))
        self.num_iter = num_iter
        self.docs = documents
        # Initialize self.z by randomly assigning each words in a document to a topic
        # After this loop, we have assigned all words to a topic.
        # So we have document-topic distribution & topic-word distribution (but it's bad since it is random)
        # We need gibbs sampling to get better distributions
        for m in range(self.M): # for each document in all documents
            N = len(documents.docs[m].docWords) # total number of words in document m
            for n in range(N): # for each word in document m
                wordIdx = self.get_word_idx(m,n)
                random_topic = np.random.randint(low=0, high=self.K) # 0 to K-1
                self.z[m].append(random_topic)
                self.MK[m][random_topic] += 1 # update the total number of the random_topic in document m
                self.KW[random_topic][wordIdx] += 1 # Update the total number of word w in the random_topic
        self.updatePhi()
        self.updateTheta()

    def fit(self):
        for i in range(self.num_iter):
            print("Iterations: " + str(i))
            for m in range(self.M): # for each document
                N = len(self.docs.docs[m].docWords)
                for n in range(N): # for each word in document m
                    new_topic_idx = self.gibbsSampling(m,n)
                    self.z[m][n] = new_topic_idx
        return

    def gibbsSampling(self, m, n):
        '''
        Sample a topic from P(z_i | z_j, w), where j != i
        returns new_topic_idx
        '''
        word_idx = self.get_word_idx(m, n)
        # remove old topic
        old_topic_idx = self.z[m][n]
        self.MK[m][old_topic_idx] -= 1
        self.KW[old_topic_idx][word_idx] -= 1
        # compute conditional probs P(z_i | z_j, w)
        probs = []
        for k in range(self.K):
            sumKW = self.get_number_of_words_in_topic(k)
            sumMK = self.get_number_of_topics(m)
            prob = ((self.KW[k][word_idx] + self.beta) / (sumKW + self.W * self.beta)
                    * (self.MK[m][k] + self.alpha) / (sumMK + self.K * self.alpha))
            probs.append(prob)
        # Normalize probs
        probs = np.array(probs) / sum(probs)
        # sample a topic from probs
        new_topic_idx = np.random.choice(self.K, p=probs)
        # update necessary info
        self.MK[m][new_topic_idx] += 1
        self.KW[new_topic_idx][word_idx] += 1
        self.updatePhi()
        self.updateTheta()
        return new_topic_idx

    # update phi: topic-word distribution
    def updatePhi(self):
        for k in range(self.K): # for each topics
                sumKW = self.get_number_of_words_in_topic(k)
                for w in range(self.W): # for each unique words
                    self.phi[k][w] = (self.KW[k][w]+self.beta) / (sumKW + self.W * self.beta)
        return

    # update theta: document-topic distribution
    def updateTheta(self):
        for m in range(self.M): # for each document
                sumMK = self.get_number_of_topics(m)
                for k in range(self.K):
                    self.theta[m][k] = (self.MK[m][k]+self.alpha) / (sumMK + self.K*self.alpha)
        return

    # some useful get functions
    def get_word_idx(self, m, n):
        '''
        returns the unique index of a word
        '''
        return self.docs.docs[m].docWords[n]

    def get_number_of_topics(self, m):
        '''
        returns the total number of topics in document m
        '''
        return sum(self.MK[m])

    def get_number_of_words_in_topic(self, k):
        '''
        returns the total number of words in topic k
        '''
        return sum(self.KW[k])

# Utility Functions
def get_indicies_of_top_n(lst, n):
    '''
    returns indices of top n values of lst
    '''
    return np.argsort(lst)[-n:]

def get_values_from_dic(dic, keys):
    '''
    returns a list of values for list of keys
    '''
    return [dic[key] for key in keys]

def get_values_from_lst(lst, indices):
    '''
    returns list of values from lst, given indices
    '''
    return [lst[idx] for idx in indices]

def get_top_n_topic_word(model, n):
    '''
    returns top n number of (word, prob) pair for each topic
    '''
    dic = {}
    keys = ['Topic '+ str(i+1) for i in range(model.K)]
    for k in range(model.K):
        top_n_indices = get_indicies_of_top_n(model.phi[k], n)
        top_n_words = get_values_from_dic(model.docs.idx2Word, top_n_indices)
        top_n_probs = [model.phi[k][idx] for idx in top_n_indices]
        word_prob = tuple(zip(top_n_words,top_n_probs))
        dic[keys[k]] = sorted(word_prob, key= lambda x: x[1], reverse=True)
    return dic

def get_top_n_document_topic(model, n):
    '''
    returns document-topic distribution
    {document1: (topic1, probs), (topic2, probs),
     document2: (topic1, probs), (topic2, probs), . . .}
    '''
    dic = {}
    keys = ['Document ' + str(i+1) + '(' + model.docs.docs[i].name +')' for i in range(model.M)]
    topics = ['topic ' + str(i+1) for i in range(n)]
    for m in range(model.M):
        top_n_indices = get_indicies_of_top_n(model.theta[m], n)
        top_n_topics = get_values_from_lst(topics, top_n_indices)
        top_n_probs = get_values_from_lst(model.theta[m], top_n_indices)
        topic_prob = tuple(zip(top_n_topics, top_n_probs))
        dic[keys[m]] = sorted(topic_prob, key = lambda x: x[1], reverse=True)
    return dic
