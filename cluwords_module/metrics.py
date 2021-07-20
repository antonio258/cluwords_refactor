import numpy as np
import scipy.spatial.distance as sci_dist
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def count_tf_idf_repr(topics, cw_words, tf_idf_t):
    cw_frequency = {}
    cw_docs = {}
    for iter_topic in topics:
        topic = iter_topic.split(' ')
        for word in topic:
            word_index = np.where(cw_words == word)[0]
            cw_frequency[word] = float(tf_idf_t[word_index].getnnz(1))
            cw_docs[word] = set(tf_idf_t[word_index].nonzero()[1])

    # n_docs = 0
    # for _cw in range(tf_idf_t.shape[0]):
    #     n_docs += float(tf_idf_t[_cw].getnnz(1))

    n_docs = tf_idf_t.data.shape[0]

    return cw_frequency, cw_docs, n_docs
    
def calc_dist_2(words, w2v_model, distance_type, t):
        l1_dist = 0
        l2_dist = 0
        cos_dist = 0
        coord_dist = 0
        t = float(t)

        for word_id1 in range(len(words)):
            for word_id2 in range(word_id1 + 1, len(words)):
                # Calcular L1 w2v metric
                l1_dist += (sci_dist.euclidean(
                    w2v_model[words[word_id1]], w2v_model[words[word_id2]]))

                # Calcular L2 w2v metric
                l2_dist += (sci_dist.sqeuclidean(
                    w2v_model[words[word_id1]], w2v_model[words[word_id2]]))

                # Calcular cos w2v metric
                cos_dist += (sci_dist.cosine(
                    w2v_model[words[word_id1]], w2v_model[words[word_id2]]))

                # Calcular coordinate w2v metric
                coord_dist += (sci_dist.sqeuclidean(
                    w2v_model[words[word_id1]], w2v_model[words[word_id2]]))

        if distance_type == 'l1_dist':
            return l1_dist / (t * (t - 1.0))
        elif distance_type == 'l2_dist':
            return l2_dist / (t * (t - 1.0))
        elif distance_type == 'cos_dist':
            return cos_dist / (t * (t - 1.0))
        elif distance_type == 'coord_dist':
            return coord_dist / (t * (t - 1.0))

        return .0
    
def get_coherence(topic, word_frequency, term_docs):
        coherence = []

        for t in range(len(topic)):
            topico = topic[t]
            top_w = topico
            #top_w = topico.split(" ")

            coherence_t = 0.0
            for i in range(1, len(top_w)):
                for j in range(0, i):
                    cont_wi = word_frequency[top_w[j]]
                    cont_wi_wj = float(
                        len(term_docs[top_w[j]].intersection(term_docs[top_w[i]])))
                    coherence_t += np.log((cont_wi_wj + 1.0) / cont_wi)

            coherence.append(coherence_t)

        return coherence

def get_pmi(topics, word_frequency, term_docs, n_docs, n_top_words):
        pmi = []
        npmi = []

        n_top_words = float(n_top_words)

        for t in range(len(topics)):
            top_w = topics[t]
            # top_w = topico.split(' ')

            pmi_t = 0.0
            npmi_t = 0.0

            for j in range(1, len(top_w)):
                for i in range(0, j):
                    ti = top_w[i]
                    tj = top_w[j]

                    c_i = word_frequency[ti]
                    c_j = word_frequency[tj]
                    c_i_and_j = len(term_docs[ti].intersection(term_docs[tj]))

                    pmi_t += np.log(((c_i_and_j + 1.0) / float(n_docs)) /
                                    ((c_i * c_j) / float(n_docs) ** 2))

                    npmi_t += -1.0 * np.log((c_i_and_j + 0.01) / float(n_docs))

            peso = 1.0 / (n_top_words * (n_top_words - 1.0))

            pmi.append(peso * pmi_t)
            npmi.append(pmi_t / npmi_t)

        return pmi, npmi
    
def get_w2v_metric(topics, t, path_to_save_model, distance_type, embedding_type=False):
        word_vectors = KeyedVectors.load_word2vec_format(
            fname='{}.txt'.format(path_to_save_model), binary=embedding_type)
        model = word_vectors.wv
        values = []

        for topic in topics:
            words = topic
            value = calc_dist_2(words, model, distance_type, t)
            values.append(value)

        return values
    
def npmi_coherence(topics, word_frequency, term_docs, n_docs, n_top_words):
    tf = word_frequency
    feature_names = term_docs
    n_documents = n_docs
    words_freq = tf.todense().T.sum(axis=1)
    #n_top_words = 101
    #words_freq[feature_names.index('sensor')]
    words_coocorrence = tf.T * tf
    words_coocorrence = words_coocorrence.toarray()
    npmi = []
    for topic in topics:
        pmi_t, npmi_t = 0.0, 0.0
        topic_words = topic
        for i in range(1, len(topic_words)):
            for j in range(0, i):
                word1 = topic_words[j] #seleciono a 1 palavra
                word2 = topic_words[i] #seleciono a 2 palavra

                word1_freq = float(words_freq[feature_names.index(word1)]) #frequencia da palavra 1
                word2_freq = float(words_freq[feature_names.index(word2)]) # frequencia da palavra 2

                word1_2freq = float(words_coocorrence[feature_names.index(word1)][feature_names.index(word2)]) #co-occorrência das palavras 1 e 2

                prob1 = word1_freq / float(n_documents) #probabilidade da palavra 1 ocorrer
                prob2 = word2_freq / float(n_documents) #probabilidade da palavra 2 ocorrer
                prob1_2 = word1_2freq / float(n_documents) #probabilidade da co-ocorrencia

                if prob1_2: 
                    pmi_t = np.log(prob1_2/ (prob1 * prob2)) #calcula o PMI
                    npmi_t += pmi_t / (-1.0 * np.log(prob1_2)) #calcula o NPMI
                    #print(pmi_t / (-1.0 * np.log(prob1_2)))
        div = 2 / (n_top_words * (n_top_words-1))
        npmi.append( div * npmi_t)

    return npmi

def npmi_equal(topics, word_frequency, term_docs, n_docs, n_top_words):
    tf = word_frequency
    feature_names = term_docs
    n_documents = n_docs
    words_freq = tf.todense().T.sum(axis=1)
    #n_top_words = 101
    #words_freq[feature_names.index('sensor')]
    words_coocorrence = tf.T * tf
    words_coocorrence = words_coocorrence.toarray()
    npmi = []
    for topic in topics:
        pmi_t, npmi_t = 0.0, 0.0
        topic_words = topic
        for i in range(1, len(topic_words)):
            for j in range(0, i):
                word1 = topic_words[j] #seleciono a 1 palavra
                word2 = topic_words[i] #seleciono a 2 palavra

                word1_freq = float(words_freq[feature_names.index(word1)]) #frequencia da palavra 1
                word2_freq = float(words_freq[feature_names.index(word2)]) # frequencia da palavra 2

                word1_2freq = float(words_coocorrence[feature_names.index(word1)][feature_names.index(word2)]) #co-occorrência das palavras 1 e 2


                pmi_t += np.log(((word1_2freq + 1.0) / float(n_documents)) /
                                        ((word1_freq * word2_freq) / float(n_documents) ** 2))

                npmi_t += -1.0 * np.log((word1_2freq + 0.01) / float(n_documents))


        npmi.append(pmi_t / npmi_t)

    return npmi