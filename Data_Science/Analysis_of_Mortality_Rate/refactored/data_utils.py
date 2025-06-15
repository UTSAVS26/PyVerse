import pandas as pd

def load_left_handed_data():
    url = "https://gist.githubusercontent.com/mbonsma/..."
    return pd.read_csv(url)

def compute_birth_year(df, survey_year=1986):
    df['Birth_year'] = survey_year - df['Age']
    return df
