import pandas as pd
import numpy as np
import os
import ast

def normaliza_valores_internos(vetor):
    # so pra assegurar que os papers com tudo zero nao fiquem zoados
    entrou = False
    for v in range(len(vetor)):
        if(vetor[v] > 0.0):
            entrou  = True
            break
    # boto que pertence ao primeiro topico
    if(entrou == False):
        vetor[0] = 1.0

    vetor = np.array(vetor)
    
    return vetor/sum(vetor) * 100

def normaliza(vetor):
    vetor = np.array(vetor)
    if(sum(vetor) == 0):
        vetor_ = vetor
    else:
        vetor_ = (vetor/sum(vetor)) * 100 
    
    return vetor_.tolist()

def normalize(lattes_link, data_pp, results_path, topicos_dominantes_file):
    # abro as relacoes o arquivo que relaciona os papers com instituições, autores...
    df = pd.read_csv(lattes_link, sep="|", dtype=str)

    data_year = pd.read_csv(data_pp, sep="|", dtype=str)

    df = df.merge(data_year[['id']], left_on='paper_id', right_on='id')
    # pra cada arquivo de topico dominante (feito para cada configuração) sobre o paper, eu vou estender essa analise para autores, paises... 
    print("Recuperando o arquivo ...")
    # abro o arquivo de topicos dominantes em questao
    df_dominante = pd.read_csv('{}/{}'.format(results_path, topicos_dominantes_file), sep = "|", dtype=str)

    # guardo o numero de topicos que eu possuo no arquivo de topicos dominantes que esta sendo gerado
    # subtrai 2 pq vem o Unnamed e o dominant_topic
    qtd_topicos = df_dominante.shape[1] - 2
    # print(qtd_topicos)
    
    papers_id = list(df['id'])
    
    # id_ = []
    id_paper_ = []
    id_author_ = []
    id_institution_ = []
    id_uf_ = []
    id_state_ = []
    topicos_ = []
    
    
    # pra cada linha do meu dataframe que veio do banco com todas as relações de id
    df_join = pd.merge(df, df_dominante,how='inner', left_on=['id'], right_on=['Unnamed: 0'])
    fim = len(list(df_join['id']))
    contador = 0

    print('running in dataframe')
    # pra cada linha de match entre o paper_id e os topicos dominantes
    for index, row in df_join.iterrows():
        #print(str(contador) + " de " + str(fim))
        contador += 1 
        # vetor com o valor dos topicos dominantes deste paper
        valores_normalizados = [0 for k in range(qtd_topicos)]
        # vetor que vai armazenar somente os topicos nao nulos do paper
        valores_positivos = []

        # pra cada topico que eu tenho neste arquivo de topicos dominantes
        for q in range(qtd_topicos):
            # eu gero o nome da coluna dele
            topico_txt = "Topico " + str(q)
            # recupero o valor de topico dominante na linha que esta sendo analisada
            valor_ = float(row[str(topico_txt)])
            # guardo o valor obtido
            valores_normalizados[q] = valor_

        # normalizo pra 100 o vetor com os valores de topico dominante pra linha que esta sendo analisada
        valores_normalizados = normaliza_valores_internos(valores_normalizados)
        # converto pra lista novamente, so pra facilitar
        valores_normalizados = valores_normalizados.tolist()

        # pra cada topico que eu tenho neste arquivo de topicos dominantes
        for q in range(qtd_topicos):
            # gero um titulo que vai armazenar o valor do topico dominante num dicionario
            topico_txt = "Topico " + str(q)
            # vou guardar somente aqueles que são positivos
            if(valores_normalizados[q] > 0.0):
                # guardo em forma de dicionario o topico dominante com valor positivo
                valores_positivos.append({str(topico_txt): valores_normalizados[q]})
                
        # neste ponto o paper que foi analisado, a linha que esta sendo comparada, possui uma lista de dicionarios com
        # {Topico Dominante: valor positivo}, no melhor dos casos ela vai ter uma lista com k dicionarios;
        # no pior, vai ter so um dicionario com o unico topico dominante dele
            
        # nesse passo adiante vou so readicionar os valores de ids que vieram da base, contudo adicionando uma lista de topicos
        # dominantes para cada id_paper
        # id_.append(row['id'])
        topicos_.append(valores_positivos)
        id_paper_.append(row['id'])
        id_author_.append(row['author_id'])
        id_institution_.append(row['work_institution']) 
        id_uf_.append(row['uf_id'])
        # id_country_.append(row['country_id'])        
    
    # com todas as linhas dos paper_ids que eu tenho no banco de dados e que eu processei tendo uma coluna com os valores de topicos dominantes
    # eu posso gerar um dataframe com esses dados - so pra visualizar o que ta sendo feito mesmo, se quiser dropar o salvamento em arquivo
    # pode ficar a vontade
    dataframe_final = pd.DataFrame()
    # dataframe_final['id'] = id_
    dataframe_final['id'] = id_paper_
    dataframe_final['author_id'] = id_author_
    dataframe_final['work_institution'] = id_institution_
    dataframe_final['uf_id'] = id_uf_
    # dataframe_final['country_id'] = id_country_
    dataframe_final['topics'] = topicos_

    print("Salvando o arquivo ...") 
    dataframe_final.to_csv('{}/partiatable_{}'.format(results_path, topicos_dominantes_file), sep="|")

    campo_id = [ 'author_id', 'work_institution', 'uf_id']
    nome_id =  [ 'authors', 'institutions', 'uf']

    # pra cada entidade que eu quero descobrir as relações de topicos dominantes, eu preciso fazer algumas manipulações na lista de dicionarios
    for cid, nid in zip(campo_id, nome_id):
    
        print("Calculando Porcentagens ...")
        # vou recuperar pra cada id da entidade que eu to ana;lisando o id correspondente e a lista de topicos
        fields=[cid, "topics"]
        df = pd.read_csv('{}/partiatable_{}'.format(results_path, topicos_dominantes_file), sep="|", usecols=fields, dtype=str)
        
        #print()
        # gero uma coluna pra cada coluna de topicos da entidade, logo se um autor aparece em varios papers como autor 
        # cada topico dominante que ele tem vai virar uma coluna
        df = (df.set_index([cid, df.groupby([cid]).cumcount()]).unstack().sort_index(axis=1, level=1))
        df.columns = ['{}_{}'.format(i, j) for i, j in df.columns]
        df = df.reset_index()

        # recupero o total de linhas agora, com os autores agrupados 
        n_linhas = len(list(df[cid]))

        # numero de colunas varia de acordo com os topicos dominantes que os autores possuem 
        colunas = len(list(df.columns)) - 1
        # print("temos " + str(colunas) + " colunas...")

        # pra cada entidade (autor, instituição, país) vamos guardar o id, os topicos normalizados por linha e os topicos normalizados por coluna
        novas_entidades_id = []
        topicos_normalizados_linha = []
        topicos_normalizados_coluna = []
        fim = len(list(df[cid]))
        inicio = 0

        # valor total dos topicos
        somatorio_topico = [0 for i in range(qtd_topicos)]

        # pra cada linha da entidade
        for index, row in df.iterrows():
            #print(str(inicio) + " de " + str(fim))
            inicio = inicio + 1

            # inicialmente um autor tem 0 em todos os topicos
            valores_topicos  = [0 for k in range(qtd_topicos)]
            
            # pra cada coluna possivel existente de topicos
            for i in range(colunas):
                # se nao for uma coluna vazia, isto é, topico vazio
                if(str(row['topics_' + str(i)]) != "nan"):
                    # print(str(row['topics_' + str(i)]))
                    # converto a representação em string da minha lista de dicionario para uma lista de dicionarios
                    lista_topicos = ast.literal_eval(str(row['topics_' + str(i)]))
                    for l in lista_topicos:
                        # pra cada dicionario eu vou recuperar o valor do topico, para isso eu preciso saber
                        # qual topico é, entao percorro todas as possibilidades
                        for n in range(qtd_topicos):
                            # vou tentar recuperar o valor de cada topico
                            try:
                                valores_topicos[n] = valores_topicos[n] + l['Topico ' + str(n)] 
                                break
                            except:
                                pass
                else:
                    # nao tem como uma coluna ser vazia e as proximas nao o serem, 
                    # logo eu iterrompo o for nessa linha
                    break

            # guardo a linha normalizada pelos valores dela mesma - logo a soma da linha da 1 (100)
            topicos_normalizados_linha.append(normaliza(valores_topicos))
            # print(valores_topicos) 
            # input()

            # vou incrementar o total de cada topico - tipo, na primeira linha, eu descobri que o topico 2 teve valor 0.5 entao eu somo agora
            # na quitna linha, o topico 2 apresenta valor 0.3, logo eu somo com os 0.5 que eu ja tinha, e assim vai
            for v in range(len(valores_topicos)):
                somatorio_topico[v] = somatorio_topico[v] + valores_topicos[v]
            # print(somatorio_topico)

            # adiciono o id da entidadde
            novas_entidades_id.append(row[cid])
            # novos_topicos.append(topicos_porcentagem)
            
            # adiciono os valores normalizados da linha, num primeiro momento, depois esses valores vao se alterar
            # de acordo com o topico global
            topicos_normalizados_coluna.append(valores_topicos)

        # segunda etapa é normalizar pelo somatorio da coluna
        for t in range(len(topicos_normalizados_coluna)):
            pos = 0
            for s in somatorio_topico:
                topicos_normalizados_coluna[t][pos] = (topicos_normalizados_coluna[t][pos] / s) * 100
                pos = pos + 1

        # vou guardar os valores de todas as colunas
        temp_colunas = [[] for k in range(qtd_topicos)]
        
        for topic in topicos_normalizados_coluna:
            # pra cada linha, que tem o campo de normalizado por coluna, eu vou recuperar o valor que eu tenho
            for n in range(qtd_topicos):
                # eu estou aqui gerando um conjunto de vetores com todos os valores, em cada vetor, de um mesmo topico
                temp_colunas[n].append(topic[n])

        # terceira etapa, vamos fazer a normalizacao pelo maximo de uma coluna entre 0 e 100
        novas_colunas = []
        #print(temp_colunas)

        for coluna in temp_colunas:
            melhores_valores = []
            melhores_entidade = []
            for c in range(len(coluna)):
                if(len(melhores_valores) < 10):
                    melhores_valores.append(coluna[c])
                    melhores_entidade.append(c)
                else:
                    for mv in range(len(melhores_valores)):
                        if (coluna[c] > melhores_valores[mv]):
                            melhores_valores[mv] = coluna[c]
                            melhores_entidade[mv] = c
                            break
            # 10 melhores
            soma_melhores = sum(melhores_valores)
            # vou zerar os caras que nao estao entre os melhores
            for c in range(len(coluna)):
                entrou = False
                for me in range(len(melhores_entidade)):
                    # achei um cara que é um dos 10 - vou normalizar
                    if( c == melhores_entidade[me] ):
                        coluna[c] = (melhores_valores[me] / soma_melhores) * 100
                        entrou = True
                        break
                # zero os caras que nao sao um dos melhores
                if(entrou == False):
                    coluna[c] = 0

            # maximo = max(coluna)
            # # print("Coluna Max:" + str(maximo))
            #
            # arr = np.array(coluna)
            #
            # arr = (arr / maximo) * 100

            novas_colunas.append(coluna)

        # agora eu vou gerar novamente o vetor de topicos normalizados por coluna para um mesmo autor, instituição ou país
        topicos_normalizados_coluna = []
        cont = 0
        # vou fazer isso, percorrendo todo o conjunto de vetores e pegando posicao a posicao e guardando na linha do autor, instituicao ou pais
        for i in range(n_linhas):
#             print(str(cont) + " de " + str(n_linhas))
            cont = cont + 1
            lista = [0 for k in range(qtd_topicos)]
            for nc in range(len(novas_colunas)):
                lista[nc] = novas_colunas[nc][i]  

            topicos_normalizados_coluna.append(lista)   
        
        df = pd.DataFrame()
        df[cid] = novas_entidades_id
        df['topics_normalized_row'] = topicos_normalizados_linha
        df['topics_normalized_column'] = topicos_normalizados_coluna
        df.to_csv("{}/normalized_topics_{}.csv".format(results_path, nid), sep="|")

