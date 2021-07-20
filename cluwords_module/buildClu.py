import argparse
from .metrics import *
from .cluwords import *
import pickle


class cluWords(object):

    def __init__(self, datapath, data_type, pp_column=None, open_pickle=None):
        self.name = datapath.split('/')[-1][:-4]
        if open_pickle:
            print("load cluwords from pickle")
            self.tfidf_cluwords = pickle.load(open('./clu_embeddings/tf_idf_cluwords_{}.p'.format(self.name), 'rb'))
            self.vocab_cluwords = pickle.load(open('./clu_embeddings/vocab_cluwords_{}.p'.format(self.name), 'rb'))
        else:
            self.tfidf_cluwords = None
            self.vocab_cluwords = None

        if data_type == 'txt':
            self.data, self.n_documents, self.ids = read_data(datapath)
        elif data_type == 'csv':
            if not pp_column:
                print("selecione uma coluna de dados no seu dataframe")
                exit()
            self.data, self.n_documents, self.ids = read_data(datapath, pp_column)
        else:
            print("tipo de arquivo inválido")
            exit()

        self.n_documents = len(self.data)
        self.n_cluwords = 0
        self.data_type = data_type


    def build_cluwords(self, embedding_file, embedding_type, k_neighbors, n_threads, threshold, save_pickle=False):

        print('building embedding...')
        words_vector, n_words = build_embedding(embedding_file, embedding_type, self.data)

        print('creating cluwords...')
        space_vector, self.vocab_cluwords = create_cluwords(words_vector)
        words_vector = None

        print('getting cosine similarity...')
        distances, indices = calcule_similarity(space_vector, k_neighbors, n_threads)

        print('filtering cluwords...')
        list_cluwords = filter_cluwords(n_words, threshold, indices, distances)
        space_vector = None
        distances, indices = None, None

        print('calculating cluwords tf...')
        cluwords_tf_idf, hyp_aux, tf, self.n_cluwords = compute_tf(n_words, self.vocab_cluwords, self.data, list_cluwords)

        print('calculating cluwords idf')
        cluwords_idf = compute_idf(hyp_aux, tf, self.n_documents)
        list_cluwords = None
        hyp_aux = None
        tf = None
        cluwords_tf_idf = cluwords_tf_idf.multiply(cluwords_idf)
        self.tfidf_cluwords = csr_matrix(cluwords_tf_idf, shape=cluwords_tf_idf.shape, dtype=np.float32)

        if save_pickle:
            os.makedirs('./clu_embeddings', exist_ok=True)
            pickle.dump(self.tfidf_cluwords, open('./clu_embeddings/tf_idf_cluwords_{}.p'.format(self.name), 'wb'))
            pickle.dump(self.vocab_cluwords, open('./clu_embeddings/vocab_cluwords_{}.p'.format(self.name), 'wb'))
    

    def get_topics(self, topics_path, n_topics, n_top_words, n_total_words, dominante_topic=False):
        print("\nFitting the NMF model (Frobenius norm) with tf-idf features, "
        "n_samples=%d and n_features=%d..." % (self.n_documents, self.n_cluwords))
        nmf = NMF(n_components=n_topics, random_state=1, alpha=.1, l1_ratio=.5, max_iter=2500).fit(self.tfidf_cluwords)
        
        topics = clean_topics(nmf, n_total_words, self.vocab_cluwords)
        if dominante_topic:
            topicos_dominantes(n_topics, nmf, self.tfidf_cluwords, topics_path, self.ids)
        save_topics(topics, n_top_words, topics_path)

    
    def calc_metrics(self, metrics, topics_path, n_top_words, dataset_embedding=None, embedding_type=None):

        if type(n_top_words) != list:
            n_top_words = [n_top_words]

        for t in n_top_words:
            with open('{}/result_topic_{}.txt'.format(topics_path, t)) as f_res:
                topics = [x.strip() for x in f_res.readlines()]
                topics = topics[2:]
                topics = [x for x in topics if x]
                f_res.close()
            cluwords_freq, cluwords_docs, n_docs = count_tf_idf_repr(topics, self.vocab_cluwords, self.tfidf_cluwords.transpose())
            topics_t = [x.split(' ') for x in topics]

            with open('{}/result_topic_{}.txt'.format(topics_path, t), 'a') as f_res:
                f_res.write('\n')
                if 'coherence' in metrics:
                    coherence = get_coherence(topics_t, cluwords_freq, cluwords_docs)
                    f_res.write('Coherence: {} ({})\n'.format(np.round(np.mean(coherence), 4), np.round(np.std(coherence), 4)))
                
                if 'npmi' in metrics:
                    pmi, npmi = get_pmi(topics=topics_t, word_frequency=cluwords_freq, term_docs=cluwords_docs, n_docs=n_docs, n_top_words=t)
                    f_res.write('avg PMI: {} ({})\n'.format(np.round(np.mean(pmi), 4), np.round(np.std(pmi), 4)))
                    #f_res.write('{}\n'.format(pmi))
                    # f_res.write('NPMI:\n')
                    # for score in npmi:
                    #     f_res.write('{}\n'.format(score))
                    f_res.write('avg NPMI: {} ({})\n'.format(np.round(np.mean(npmi), 4), np.round(np.std(npmi), 4)))

                if 'w2v_l1' in metrics:
                    if dataset_embedding and embedding_type:
                        w2v_l1 = get_w2v_metric(topics, t, dataset_embedding, 'l1_dist')
                        f_res.write('W2V-L1: {} ({})\n'.format(np.round(np.mean(w2v_l1), 4), np.round(np.std(w2v_l1), 4)))
                        f_res.write('{}\n'.format(w2v_l1))
                    else:
                        print("Defina o caminho do embedding e o tipo")


def run(datapath, embedding_path, n_topics, results_path, threshold=0.4, metrics=None, n_top_words=None):
    """
    Executa as cluwords
    Args:
        datapath: diretório do arquivo de entrada
        embedding_path: diretório do embedding
        n_topics: quantidade de tópicos
        results_path: diretório de saída dos resultados
        threshold: limiar de vizinhança das cluwords (default 0.4)
        metrics: métricas de avaliação (default ['npmi']
        n_top_words: número de palavras por tópicos (default [10, 20, 30]

    Returns:

    """
    N_TOTAL_WORDS = 101
    K_NEIGHBORS = 500
    N_THREADS = 4

    if '.txt' in datapath:
        data_type = 'txt'
    elif '.csv' in datapath:
        data_type = 'csv'
    else:
        print('tipo de arquivo inválido')
        exit()

    if '.bin' in embedding_path:
        embedding_bin = True
    else:
        embedding_bin = False

    if not metrics:
        metrics = ['npmi']

    if not n_top_words:
        n_top_words = [10, 20, 30]

    cluwords = cluWords(datapath=datapath, data_type=data_type)
    cluwords.build_cluwords(embedding_file=embedding_path, embedding_type=embedding_bin, k_neighbors=K_NEIGHBORS, n_threads=N_THREADS,
        threshold=threshold)
    cluwords.get_topics(topics_path=results_path, n_topics=n_topics, n_top_words=n_top_words, n_total_words=N_TOTAL_WORDS)
    cluwords.calc_metrics(topics_path=results_path, metrics=metrics, n_top_words=n_top_words)


def run_console():
    parser = argparse.ArgumentParser(description='Run cluwords and topic modelling.')
    parser.add_argument('-d', action='store', dest='data', required=True, help='datapath')
    parser.add_argument('-e', action='store', dest='embedding', required=True, help='embedding path')
    parser.add_argument('-n', action='store', dest='ntopics', required=True, help='number of topics')
    parser.add_argument('-r', action='store', dest='results', required=True, help='results path')
    
    args = parser.parse_args()
    N_TOTAL_WORDS = 101
    K_NEIGHBORS = 500
    N_THREADS = 4
    THRESHOLD = 0.4

    if '.txt' in args.data:
        data_type = 'txt'
    elif '.csv' in args.data:
        data_type = 'csv'
    else:
        print('tipo de arquivo inválido')
        exit()

    if '.bin' in args.embedding:
        embedding_bin = True
    else:
        embedding_bin = False

    cluwords = cluWords(datapath=args.data, data_type=data_type)
    cluwords.build_cluwords(embedding_file=args.embedding, embedding_type=embedding_bin, k_neighbors=K_NEIGHBORS, n_threads=N_THREADS,
        threshold=THRESHOLD)
    cluwords.get_topics(topics_path=args.results, n_topics=args.ntopics, n_top_words=[10, 20, 30], n_total_words=N_TOTAL_WORDS)
    cluwords.calc_metrics(topics_path=args.results, metrics=['npmi'], n_top_words=[10, 20, 30])
