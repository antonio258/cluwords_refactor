from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords as sw
import pandas as pd
import glob
import os
from shutil import copyfile


def hash_siglas(papers, sigla_path, pontuacao_path):
    arquivo = open(sigla_path, "w", encoding="utf-8")
    tokenizador = TweetTokenizer()
    stopwords = set(sw.words('english'))
    
    
    arquivo2 = open(pontuacao_path, "r", encoding="utf-8")
    linhas = arquivo2.readlines()
    arquivo2.close()

    pontuaction = []
    for linha in linhas:
        # tiro o \n
        p = linha.split("\n")[0]
        pontuaction.append(p)
    
    for paper in papers:
        tokens = tokenizador.tokenize(paper.lower())
        for i in range(len(tokens)):
            # acho o começo dos parenteses
            if(tokens[i] == "("):
                # percorro a partir do inicio do parenteses
                for j in range(i, len(tokens)):
                    # até achar o fim do parenteses
                    if(tokens[j] == ")"):
                        # quando eu acho eu printo o conteudo do que esta entre parenteses, menos os parenteses
                        # se eu possuir somente um elemento entre parenteses, quer dizer que temos uma sigla
                        if((j - i - 1== 1) and (len(tokens[i + 1]) >= 2) and (len(tokens[i + 1]) <= 4) and (str(tokens[i+1]).isdecimal() == False)):
                            sigla = tokens[i + 1]
                            # print(sigla)

                            significado = []
                            cont = 1
                            tamanho = len(sigla) + 1
                            # faço um loop ao contrario para escrever a sigla corretamente
                            while(cont < tamanho):
                                if((str(tokens[i - cont]) not in stopwords) and (str(tokens[i - cont]) not in pontuaction)):
                                    # print(tokens[i - cont])
                                    significado.append(tokens[i - cont])
                                
                                # se tiver um hifen, provavelmente ele ja e considerado como inicial da sigla
                                if("-" in tokens[i - cont]):
                                    # por isso eu ja avanço ("volto") nas palavras que definem a sigla
                                    cont = cont + 1   

                                # aumento o contador pra decrescer
                                cont = cont + 1

                            significado_sigla = ""
                            for t in range(len(significado) - 1, -1, -1):
                                # a ultima palavra nao precisa do _ no final
                                if(t == 0):
                                    significado_sigla = significado_sigla + str(significado[t]).replace("-", "_")                            
                                else:
                                    significado_sigla = significado_sigla + str(significado[t]).replace("-", "_") + "_"
                            
                            temp = significado_sigla + ">>>>>" + sigla + "\n"
                            arquivo.write(temp)    
                        # interrompo o loop para achar o fim do parenteses, quem determina o avanço no texto
                        # é a abertura de parenteses
                        break
        
    arquivo.close()
    
    
def read_data(data_path):
    df_papers = pd.read_csv(data_path, sep='|', dtype=str)
    df_papers['title'].fillna("", inplace = True)
    df_papers['abstract'].fillna("", inplace = True)
    title = df_papers['title'].values
    abstract = df_papers['abstract'].values
    data = title + ". " + abstract
    return data

def sigla2topic(sigla_path, topic_path, save_path):
    with open(sigla_path, 'r') as file:
        siglas = file.read().split('\n')
    with open(topic_path, 'r') as file:
        topics = file.readlines()
    
    sg = {}
    for sigla in siglas:
        try:
            sig = sigla.split('>>>>>')[1]
            complete = sigla.split('>>>>>')[0]
            sg[sig] = complete
        except:
            pass
    
    onlytopics = []
    for line in topics[2:]:
        if 'NPMI' in line:
            break
        else:
            onlytopics.append(line)
            
    onlytopics = [t.split(' ') for t in onlytopics]
    n_topics = len(onlytopics)       
    
    for index1, topic in enumerate(onlytopics):
        for index2, word in enumerate(topic):
            if word in sg:
                onlytopics[index1][index2] = sg[word]
    onlytopics = [" ".join(x) for x in onlytopics]
    with open(save_path, 'w') as file:
        file.write(str(topics[0]))
        file.write(str(topics[1]))
        for topic in onlytopics:
            file.write(topic)
        for topic in topics[n_topics+2:]:
            file.write(topic)

def get_siglas(datapath, sigla_path, topic_path, result_path, pontuacao_path):    
    sig_topic = result_path
    os.makedirs(sig_topic, exist_ok=True)
    data = read_data(datapath)
    hash_siglas(data, sigla_path, pontuacao_path)
    
    for file in glob.glob(topic_path+'*'):
        if 'result_topic' in file:
            save_path = sig_topic + '/' + file.split('/')[-1:][0]
            sigla2topic(sigla_path, file, save_path)
        else:
            copyfile(file, sig_topic + file.split('/')[-1:][0])
    
def main():
    datapath = '../data/years/1966_2000.csv'
    sigla_path = "../data/hash_siglas.txt"
    topic_path = '../resultados/cluwords/com_ngrams/to_banco/1966_2000size_50_window_12_iter_12_th_0.4/'
    sig_topic = '../resultados/cluwords/com_ngrams/to_banco/1966_2000size_50_window_12_iter_12_th_0.4_sig/'
    os.makedirs(sig_topic, exist_ok=True)
    data = read_data(datapath)
    hash_siglas(data, sigla_path)
    
    for folder in glob.glob(topic_path+'*'):
        if '.csv' not in folder:
            foldername = folder.split('/')[-1:][0]
            print(foldername)
            os.makedirs(sig_topic + foldername, exist_ok=True)
            for file in glob.glob(topic_path + foldername + '/*'):
                if 'result_topic' in file:
                    save_path = sig_topic + foldername + '/' + file.split('/')[-1:][0]
                    print(file)
                    sigla2topic(sigla_path, file, save_path)
                else:
                    copyfile(file, sig_topic + foldername + '/' + file.split('/')[-1:][0])
        else:
            copyfile(folder, sig_topic + folder.split('/')[-1:][0])
    
if __name__ == "__main__":
    main()
