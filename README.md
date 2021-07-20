# cluwords_refactor
Modificações nos scripts das cluwords

## Execução

* navegue até o diretório cluwords_refactor

```pip install .```

* Carregar bibliotecas
```from cluwords_module import run, , fasttext_embedding```

* construção do embedding

```fasttext_embeddind(datapath='../../data.txt', embedding_path='../teste_embedding.bin')```

* Modelagem de tópicos

```run(datapath='../../data.txt', embedding_path='../../wiki-news-300d-1M.vec', n_topics=10, results_path='resultados/teste_all')```
