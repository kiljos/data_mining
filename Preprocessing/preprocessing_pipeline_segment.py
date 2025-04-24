from preprocessing_pipeline_initial import preprocessing_pipeline
import pandas as pd


def add_segment_to_df():
    df = preprocessing_pipeline()
    segments = pd.read_csv('df_mit_segment.csv')
    df = pd.merge(df, segments, how='left', on = ['brand', 'model'])
    if df['segment'].isnull().sum() > 0: 
        raise ValueError("There are missing values in the 'segment' column after merging.")
    else: 
        return df
    
