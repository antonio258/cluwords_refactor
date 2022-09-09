from gensim.models import Word2Vec

def word2vec_embedding(datapath, embedding_path, vector_size=100, window=10, epoch=10, workers=4, sg=1, binary=True, return_model=False):
    if type(datapath) == str:
       with open(datapath) as df:
           data = df.read().splitlines()
           df.close()
       data = [x.split(' ') for x in data]
       model = Word2Vec(sentences=data, vector_size=vector_size, window=window, workers=workers, sg=sg, epochs=epoch)
    else:
        data = [x.split(' ') for x in data]
        model = Word2Vec(sentences=data, vector_size=vector_size, window=window, workers=workers, sg=sg, epochs=epoch)
        
    model.wv.save_word2vec_format(embedding_path, binary=binary)
    if return_model:
        return model
    del model
