# import sys
# assert sys.version_info[0]==3
# assert sys.version_info[1] >= 5

# from gensim.models import KeyedVectors
# from gensim.test.utils import datapath
# import pprint
# import matplotlib.pyplot as plt
# plt.rcParams['figure.figsize'] = [10, 5]
# import nltk
# nltk.download('reuters')
# from nltk.corpus import reuters
# import numpy as np
# import random
# import scipy as sp
# from sklearn.decomposition import TruncatedSVD
# from sklearn.decomposition import PCA

# START_TOKEN = '<START>'
# END_TOKEN = '<END>'

# np.random.seed(0)
# random.seed(0)

def read_corpus(category="crude"):
    """ Read files from the specified Reuter's category.
        Params:
            category (string): category name
        Return:
            list of lists, with words from each of the processed files
    """
    files = reuters.fileids(category)
    return [[START_TOKEN] + [w.lower() for w in list(reuters.words(f))] + [END_TOKEN] for f in files]

def distinct_words(corpus):
    """ Determine a list of distinct words for the corpus.
        Params:
            corpus (list of list of strings): corpus of documents
        Return:
            corpus_words (list of strings): list of distinct words across the corpus, sorted (using python 'sorted' function)
            num_corpus_words (integer): number of distinct words across the corpus
    """
    corpus_words = []
    num_corpus_words = -1
    
    # ------------------
    # Write your implementation here.
    corpus_words =  sorted(list(set([word for sentence in corpus for word in sentence])))
    num_corpus_words = len(corpus_words)

    # ------------------

    return corpus_words, num_corpus_words

def compute_co_occurrence_matrix(corpus, window_size=4):
    """ Compute co-occurrence matrix for the given corpus and window_size (default of 4).
    
        Note: Each word in a document should be at the center of a window. Words near edges will have a smaller
              number of co-occurring words.
              
              For example, if we take the document "START All that glitters is not gold END" with window size of 4,
              "All" will co-occur with "START", "that", "glitters", "is", and "not".
    
        Params:
            corpus (list of list of strings): corpus of documents
            window_size (int): size of context window
        Return:
            M (numpy matrix of shape (number of corpus words, number of corpus words)): 
                Co-occurence matrix of word counts. 
                The ordering of the words in the rows/columns should be the same as the ordering of the words given by the distinct_words function.
            word2Ind (dict): dictionary that maps word to index (i.e. row/column number) for matrix M.
    """
    words, num_words = distinct_words(corpus)
    M = np.zeros((num_words, num_words))
    word2Ind = dict([(word, index) for index, word in enumerate(words)])
    
    # ------------------
    # Write your implementation here.
    for sentence in corpus:
        current_index = 0
        sentence_len = len(sentence)
        indices = [word2Ind[i] for i in sentence]
        while current_index < sentence_len:
            left  = max(current_index - window_size, 0)
            right = min(current_index + window_size + 1, sentence_len) 
            current_word = sentence[current_index]
            current_word_index = word2Ind[current_word]
            words_around = indices[left:current_index] + indices[current_index+1:right]
            
            for ind in words_around:
                M[current_word_index, ind] += 1
            
            current_index += 1

    # ------------------

    return M, word2Ind

def reduce_to_k_dim(M, k=2):
    """ Reduce a co-occurence count matrix of dimensionality (num_corpus_words, num_corpus_words)
        to a matrix of dimensionality (num_corpus_words, k) using the following SVD function from Scikit-Learn:
            - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
    
        Params:
            M (numpy matrix of shape (number of corpus words, number of corpus words)): co-occurence matrix of word counts
            k (int): embedding size of each word after dimension reduction
        Return:
            M_reduced (numpy matrix of shape (number of corpus words, k)): matrix of k-dimensioal word embeddings.
                    In terms of the SVD from math class, this actually returns U * S
    """    
    n_iters = 10     # Use this parameter in your call to `TruncatedSVD`
    M_reduced = None
    print("Running Truncated SVD over %i words..." % (M.shape[0]))
    
    # ------------------
    # Write your implementation here.
    TSVD = TruncatedSVD(n_components=k, n_iter=n_iters)
    M_reduced = TSVD.fit_transform(M)

    # ------------------

    print("Done.")
    return M_reduced

def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    print('load api success')
    vocab = list(wv_from_bin.vocab.keys())
    print(len(vocab))
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin

def get_matrix_of_vectors(wv_from_bin, required_words=['barrels', 'bpd', 'ecuador', 'energy', 'industry', 'kuwait', 'oil', 'output', 'petroleum', 'venezuela']):
    """ Put the word2vec vectors into a matrix M.
        Param:
            wv_from_bin: KeyedVectors object; the 3 million word2vec vectors loaded from file
        Return:
            M: numpy matrix shape (num words, 300) containing the vectors
            word2Ind: dictionary mapping each word to its row number in M
    """
    import random
    words = list(wv_from_bin.vocab.keys())
    print("Shuffling words ...")
    random.shuffle(words)
    words = words[:10000]
    print("Putting %i words into word2Ind and matrix M..." % len(words))
    word2Ind = {}
    M = []
    curInd = 0
    for w in words:
        try:
            M.append(wv_from_bin.word_vec(w))
            word2Ind[w] = curInd
            curInd += 1
        except KeyError:
            continue
    for w in required_words:
        try:
            M.append(wv_from_bin.word_vec(w))
            word2Ind[w] = curInd
            curInd += 1
        except KeyError:
            continue
    M = np.stack(M)
    print("Done.")
    return M, word2Ind

wv_from_bin = load_word2vec()
M, word2Ind = get_matrix_of_vectors(wv_from_bin)
print("M:", M)
M_reduced = reduce_to_k_dim(M, k=2)
print("M_reduced:", M_reduced)
words = ['barrels', 'bpd', 'ecuador', 'energy', 'industry', 'kuwait', 'oil', 'output', 'petroleum', 'venezuela']
for word in words:
    index = word2Ind[word]
    embedding = M_reduced[index]
    print("embedding:", embedding)