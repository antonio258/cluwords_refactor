from gensim.models import FastText


def fasttext_embedding(datapath, embedding_path, vector_size=100, window=10, epoch=10, workers=4, sg=1, binary=True):
    with open(datapath) as f:
        data = [x.strip() for x in f.readlines()]
    model = FastText(vector_size=vector_size, window=window, sentences=data, epochs=epoch, sg=sg, workers=workers)
    model.wv.save_word2vec_format(embedding_path, binary=binary)
    del model
