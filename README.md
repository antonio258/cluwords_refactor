# cluwords_refactor
Modificações nos scripts das cluwords

## Execução

* navegue até o diretório cluwords_refactor

```pip install .```

* Carregar bibliotecas
```from tm_module import cluWords```

* Modelagem de tópicos

```
df = pd.read_parquet('./cluwords_refactor/2022-08-09.parquet')[['pp_text', 'id']].head(1000)
cluwords = cluWords(data=df, pp_column='pp_text', id_column='id')

cluwords.build_cluwords(
    embedding_file='/mnt/HD/embeddings/embedding_blog_tweet_w_5_e_5.vec',
    embedding_type=False,
    k_neighbors=100,
    n_threads=12,
    threshold=0.4
)

cluwords.get_topics(
    topics_path='../exemplo/',
    n_topics=10,
    n_top_words=[10],
    n_total_words=101,
    dominante_topic=True
)
```
