from jtt.preprocess import *

import pandas as pd
import numpy as np
import dask.dataframe as dd



def preprocess_dataset(df, normalization_function=z_score_normalisation):

    cnt_features = 256

    feat_columns = [col for col in df.columns if col != 'id_job']

    for col in feat_columns:

        new_df = df[col].apply(lambda x: pd.Series(str(x).split(",")).astype('int'),
                               meta = dict.fromkeys(range((cnt_features+1)), 'int')).compute()
        feature_id = new_df.loc[0,0]
        new_df.drop(columns=[0], inplace=True)

        new_df = normalization_function(new_df)
        columns_name = ['feature_' + str(feature_id) + f'_{str(i)}' for i in range(cnt_features)]
        new_df.columns = columns_name

        new_df[f'max_feature_{feature_id}_index'] = max_feature_index(new_df[columns_name])
        new_df[f'max_feature_{feature_id}_abs_mean_diff'] = max_feature_abs_mean_diff(new_df[columns_name])
        
    return new_df


def main():

    path = '/data/test.tsv'
    data = dd.read_csv(path, sep='\t', )
    df = preprocess_dataset(data)
    df.to_csv('/data/test_processed.tsv', sep='\t')


main()
