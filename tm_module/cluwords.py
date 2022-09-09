from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
from pyjarowinkler import distance
from gensim.models.phrases import Phraser, Phrases
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from sklearn.feature_extraction.text import TfidfVectorizer
from .norm_entity import norm_entity
import pandas as pd
import numpy as np
import os
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import sys

def read_data(datapath, pp_column=None):
    if pp_column:
        df_papers = pd.read_csv(datapath, sep='|', dtype=str)
        df_papers = df_papers.replace({np.nan: ''})
        df_papers = df_papers.replace({'None': ''})
        try:
            ids = df_papers['id'].values
        except:
            ids = df_papers.index
        data = df_papers[pp_column].values

    else:
        with open(datapath) as f:
            data = f.readlines()
        data = [x.strip() for x in data]
        ids = range(len(data))
    
    n_documents = len(data)
    return data, n_documents, ids

def tfidf(data):
    vectorizer = TfidfVectorizer(encoding='utf-8', 
                                 analyzer='word', 
                                 use_idf=True, 
                                 smooth_idf=False, 
                                 stop_words=None,
                                 tokenizer=lambda x: x.split(' '))
    X = vectorizer.fit_transform(data)
    return X, vectorizer.get_feature_names()

def only_tf(data):
    tf_vectorizer = CountVectorizer(
        max_df=0.95, min_df=2, tokenizer=lambda x: x.split(' ')
    )
    tf = tf_vectorizer.fit_transform(data)
    return tf, tf_vectorizer.get_feature_names()


def build_embedding(embedding_file, embedding_bin, data):
        model = KeyedVectors.load_word2vec_format(embedding_file, binary=embedding_bin)
        vec = CountVectorizer()
        vec.fit(data)
        dataset_words = vec.get_feature_names()
        model_words = model.index_to_key
        intersection_vocab = set(dataset_words).intersection(model_words)
        words_vector = {word: model[word] for word in intersection_vocab}
        
        # for word in dataset_words:
        #     try:
        #         words_vector[word] = model[word]         
        #     except KeyError:
        #         continue
        del model
        n_words = len(words_vector)
        total_words = len(dataset_words)
        print('Number of cluwords {}'.format(n_words))
        # os.makedirs('../embeddings', exist_ok=True)
        # file = open("{}.txt".format(dataset_embedding), 'w')
        # file.write('{} {}\n'.format(n_words, str(embedding_size)))
        # for index, word_vec in words_vector.items():
        #     file.write("%s\n" % str(index + ' ' + " ".join([str(round(x, 9)) for x in word_vec.tolist()])))
        # file.close()
        return words_vector, n_words, f"Percent of vocabulary in embedding: {round(n_words/total_words, 2)}"


def create_cluwords(words_vector):
    space_vector = [np.array([round(y,9) for y in words_vector[x].tolist()]) for x in words_vector]
    space_vector = np.array(space_vector)
    vocab_cluwords = np.asarray([x for x in words_vector])
    return space_vector, vocab_cluwords


def calcule_similarity(space_vector, k_neighbors, n_threads):
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='auto', metric='cosine', n_jobs=n_threads).fit(space_vector)
    distances, indices = nbrs.kneighbors(space_vector)
    return distances, indices


def filter_cluwords(n_words, threshold, indices, distances):
    list_cluwords = np.zeros((n_words, n_words), dtype=np.float16)
    if threshold:
        for p in range(0, n_words):
            for i, k in enumerate(indices[p]):
                # .875, .75, .625, .50
                if 1 - distances[p][i] >= threshold:
                    list_cluwords[p][k] = round(1 - distances[p][i], 2)
                else:
                    list_cluwords[p][k] = 0.0
    else:
        for p in range(0, n_words):
            for i, k in enumerate(indices[p]):
                list_cluwords[p][k] = round(1 - distances[p][i], 2)
    return list_cluwords


def divide_array(array1, array2):
    if array1.shape[0] != array2.shape[0] or array2.shape[1] != array2.shape[1]:
        print('Tamando de tabelas inconsistentes: {} - {}'.format(array1.shape, array2.shape))
    result = np.zeros(array1.shape)
    for i in range(0, array1.shape[0]):
        for j in range(0, array1.shape[1]):
            if array2[i, j]:
                result[i][j] = array1[i, j] / array2[i, j]
    return result


def compute_tf(n_words, vocab_cluwords, data, list_cluwords):
    tf_vectorizer = CountVectorizer(max_features=n_words, binary=False, vocabulary=vocab_cluwords, tokenizer=lambda x: x.split(' '))
    tf = csr_matrix(tf_vectorizer.fit_transform(data), dtype=np.float16)
    n_cluwords = len(vocab_cluwords)
    print('tf shape {}'.format(tf.shape))
    hyp_aux = []
    for w in range(0, n_cluwords):
        hyp_aux.append(np.asarray(list_cluwords[w], dtype=np.float16))
    hyp_aux = np.asarray(hyp_aux, dtype=np.float16)
    hyp_aux = csr_matrix(hyp_aux, shape=hyp_aux.shape, dtype=np.float16)  # ?test sparse matrix!

    cluwords_tf_idf = tf.dot(hyp_aux.transpose())
    print(cluwords_tf_idf.shape)
    return cluwords_tf_idf, hyp_aux, tf, n_cluwords


def compute_idf(hyp_aux, tf, n_documents):
    #hyp_aux = hyp_aux.todense()

    #print('Dot tf and hyp_aux')
    #_dot = np.dot(tf, np.transpose(hyp_aux))  # np.array n_documents x n_cluwords # Correct!
    _dot = tf.dot(hyp_aux.transpose())
    #_dot = tf.dot(hyp_aux)

    print('Divide hyp_aux by itself')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        #bin_hyp_aux = csr_matrix(np.nan_to_num(np.divide(hyp_aux, hyp_aux))) #binary, mostra quais termos uma cluword possui
        # hyp_aux_bin = hyp_aux.astype(np.float32).toarray()
        # hyp_aux_bin = np.where(hyp_aux_bin > 0, 1, hyp_aux_bin)
        # hyp_aux_bin = csr_matrix(hyp_aux_bin, dtype=np.float16)
        hyp_aux_bin = csr_matrix((hyp_aux > 0) * 1., dtype=np.float32)

    print('Dot tf and bin hyp_aux')

    _dot_bin = tf.dot(hyp_aux_bin.transpose()) #calcula o número de termos de uma cluword em cada documento: soma(tf * term_clu (0-1))
    _dot_bin.data = 1 / _dot_bin.data
    #n_termos_cluwords por documento

    #_dot = _dot.todense()
    #_dot_bin = _dot_bin.todense()
    print('Divide _dot and _dot_bin')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        #mu_hyp = csr_matrix(np.nan_to_num(np.divide(_dot, _dot_bin))) #o somário dos valores de cada cluword divido pelo quantidade da cluword em um documento
        #mu_hyp = np.divide(_dot, dot_bin)

        mu_hyp = _dot.multiply(_dot_bin)
        print(mu_hyp.shape)
        mu_hyp = np.nan_to_num(mu_hyp, copy=False)
        # mu_hyp = csr_matrix(mu_hyp)
        #mu_hyp = csr_matrix(np.nan_to_num(np.divide(_dot, _dot_bin)))
        #cluword / n_termos de uma cluword em um documento

    print('Sum')
    cluwords_idf = np.sum(mu_hyp, axis=0) # o somátorio d

    cluwords_idf[cluwords_idf == .0] = 0.001
    print('log')
    cluwords_idf = np.log10(np.divide(n_documents, cluwords_idf)) # calcula o idf
    return cluwords_idf


def clean_topics(nmf, n_top_words, vocab_cluwords):
    topics = []
    for topic in nmf.components_:
        top = ''
        top += ' '.join([vocab_cluwords[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]])
        topics.append(str(top))

    topics_t = []
    for topic in topics:
        topic_t = topic.split(' ')
        topic_t = [x for x in topic_t if x and len(x) > 1]
        topics_t.append(topic_t)
    topics = topics_t
    topics_t = []
    for topic in topics:
        filtered_topic = []
        insert_word = np.ones(len(topic))
        for w_i in range(0, len(topic)-1):
            if insert_word[w_i]:
                filtered_topic.append(topic[w_i])
                for w_j in range((w_i + 1), len(topic)):
                    if distance.get_jaro_distance(topic[w_i], topic[w_j], winkler=True, scaling=0.1) > 0.75:
                        insert_word[w_j] = 0

        topics_t.append(filtered_topic)
    topics = topics_t
    return topics



def topicos_dominantes(n_topicos, model, matriz_tfidf, path, data, ids, id_column='', entities=[]):
    # lista com o total de topicos
    topicnames = ['Topico ' + str(i) for i in range(model.n_components)]
    # lista com todos os papers
    papernames = [str(i) for i in ids]
    # cria um dataframe onde as linhas sao os papers e as colunas sao os topicos
    print(matriz_tfidf.shape, len(papernames))
    df_document_topic = pd.DataFrame(np.round(model.transform(matriz_tfidf),n_topicos), columns=topicnames)
    df_document_topic[id_column if id_column else 'id'] = papernames
    #df_document_topic = pd.DataFrame()
    #df_document_topic['paper_id'] = np.round(model.transform(matriz_tfidf),n_topicos)

    # cria um atributo pra ver qual é o topico dominante de cada paper
    df_document_topic['dominant_topic'] = np.argmax(df_document_topic.drop(id_column if id_column else 'id', axis=1).values, axis=1)

    # print(df_document_topic['dominant_topic'])
    sns.countplot(x=df_document_topic.dominant_topic)
    plt.savefig(path + "Topicos_Dominantes.png")
    plt.close()

    df_document_topic.to_csv(path + "Topicos_Dominantes.csv", sep="|")
    resumo = pd.DataFrame()
    resumo['papers'] = papernames
    resumo['dominant_topic'] = df_document_topic['dominant_topic'].values
    resumo.to_csv(path + "Resumo_Topicos_Dominantes.csv", sep="|", index=False)

    df_copy = df_document_topic.copy().reset_index(drop=True)
    data = data.reset_index(drop=True)

    if entities:
        norm_entity(df_copy, data, id_column, entities, path)


def save_topics(topics, n_top_words, results_path, vocabulary_embedding = ''):
    if type(n_top_words) != list:
        n_top_words = [n_top_words]
    os.makedirs(results_path, exist_ok=True)
    
    for t in n_top_words:
        n_topics = len(topics)
        with open('{}/result_topic_{}.txt'.format(results_path, t), 'w') as f_res:
            f_res.write('Topics {} N_Words {} - {}\n'.format(n_topics, t, vocabulary_embedding))
            f_res.write('Topics:\n')
            topics_t = []
            for topic in topics:
                topics_t.append(topic[:t])
                for word in topic[:t]:
                    f_res.write('{} '.format(word))

                f_res.write('\n')
            f_res.close()