import pandas as pd
import numpy as np
import ast
from tqdm import tqdm

def find_null(data):
    data[np.isnan(data)] = 0
    return data

def norm_col(data):
    l = data.values
    l = [arr.tolist() for arr in l]
    l = np.array(l)
    x_normed = l / l.sum(axis=1, keepdims=1)
    x_normed = [element * 100 for element in x_normed]
    return x_normed

def normalize(data):
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    data = data / sum(data)
    data = 100 * data
    data = np.round(data,2)
    return data

def sum_array(data):
    return sum(data)



def norm_entity(df_topics, data, id_column, entities, save_path):
    cols = list(df_topics.columns)
    cols.remove('dominant_topic')
    cols.remove(id_column)
    cols_array = df_topics[cols].values.tolist()
    df_topics['array'] = cols_array
    df_topics.rename(columns={'dominant_topic': 'dmnt'}, inplace=True)
    df = df_topics[[id_column, 'array', 'dmnt']].copy().reset_index(drop=True)
    data = data.reset_index(drop=True)

    df['topics'] = df['array'].apply(normalize)
    df['sum'] = df['topics'].apply(sum_array)
    df[df['sum'] > 0].to_parquet(f"{save_path}/normalized_topics.parquet")

    df = pd.merge(data, df, left_index=True, right_index=True)
    # df = pd.merge(data, df, on='id')


    for entity in entities:
        df_filter = df[df['topics'].apply(max) >= 10]
        df_entity = df_filter.groupby(entity).agg({'topics':sum})
        df_entity.rename(columns={'topics': 'ranks'}, inplace=True)

        df_entity['ranks'] = df_entity['ranks'].apply(find_null)

        df_entity['ranks_norm'] = norm_col(df_entity.ranks)
        df_entity['sum'] = df_entity['ranks_norm'].apply(sum_array)

        df_entity.reset_index(inplace=True)
        df_entity[df_entity['sum'] > 0].to_parquet(f"{save_path}/normalized_topics_{entity}.parquet")