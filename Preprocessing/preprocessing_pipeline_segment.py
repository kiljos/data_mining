import pandas as pd


def preprocessing_pipeline_segment(df, path = '../../../df_mit_segment.csv'):
    segments = pd.read_csv(path)
    df = pd.merge(df, segments, how='left', on = ['brand', 'model'])
    if df['segment'].isnull().sum() > 0: 
        raise ValueError("There are missing values in the 'segment' column after merging.")
    else: 
        return df
    