from gensim.models import FastText


def fasttext_embedding(datapath, embedding_path, vector_size=100, window=10, epoch=10, workers=4, sg=1, binary=True, return_model=False):
    if type(datapath) == str:
        model = FastText(vector_size=vector_size, window=window, workers=workers)
        model.build_vocab(corpus_file=datapath)
        model.train(corpus_file=datapath, epochs=epoch,
                    total_examples=model.corpus_count, total_words=model.corpus_total_words)
    else:
        model = FastText(vector_size=vector_size, window=window, workers=workers)
        model.build_vocab(sentences=datapath)
        model.train(sentences=datapath, total_examples=len(datapath), epochs=epoch)
    # model = FastText(vector_size=vector_size, window=window, sentences=data, epochs=epoch, sg=sg, workers=workers)
    model.wv.save_word2vec_format(embedding_path, binary=binary)
    if return_model:
        return model
    del model
