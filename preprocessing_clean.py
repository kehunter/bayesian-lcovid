import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer


def preprocess_simple_model(df_full):
    feat_names = ["AGEP_A", "SEX_A", "SHTCVD191_A", "LONGCOVD1_A"]
    y_name = 'LONGCOVD1_A'
    
    # subset for features and drop null values
    df = df_full.loc[:, feat_names].dropna(subset=y_name)
    df = df[df[y_name] != 9]
    print("Shape after feature selection and dropping lcovid nulls:",df.shape)
    
    # FIX ATTRIBUTES
    df["SEX_A"] = (df["SEX_A"] == 1) * 1  # 1 is male 
    df["SHTCVD191_A"] = (df["SHTCVD191_A"] == 1) * 1  # 1 is yes covid shot
    
    # split X and y
    X = df.loc[:, df.columns != y_name]
    y = df.loc[:, y_name]
    y = (y == 1) * 1 # encode as 0 / 1 values
    
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=207)
    return X_train, X_test, y_train, y_test


def preprocess_lit_features(df_full):
    """Preprocess features from the literature"""
    y_name = 'LONGCOVD1_A'
    lit_feats = ["SEX_A", "DIBEV_A", "AGEP_A", "BMICAT_A", 
                 "COGMEMDFF_A", "SHTCVD191_A", "SHTCVD19NM1_A", 
                 "SHOTTYPE2_A", "HISPALLP_A"]
    df_lit = df_full[lit_feats].fillna(-1)
    df_lit[y_name] = df_full[y_name]
    df_lit = df_lit.dropna(subset=y_name)
    df_lit = df_lit[df_lit[y_name] != 9]
    df_lit.loc[:,"AGEP_A"].loc[df_lit.loc[:,"AGEP_A"] > 84] = 85
    
    encoder = OneHotEncoder(drop='first', sparse_output=False).set_output(transform="pandas")
    scaler = StandardScaler().set_output(transform="pandas")
    
    X = df_lit.drop(columns=y_name)
    
    # define continuous and categorical features
    continuous = X.columns[X.apply(np.unique).apply(len) > 15] 
    categorical = X.columns[~X.columns.isin(continuous)]
    
    # scale and encode
    cont_df =  scaler.fit_transform(X.loc[:, continuous])
    cat_df = encoder.fit_transform(X.loc[:,categorical])
    
    processed_df = pd.concat((cont_df, cat_df), axis=1)
    
    X = processed_df.copy()
    y = df_lit.loc[:, y_name]
    y = (y == 1) * 1 # encode as 0 / 1 values
    
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=207)
    return X_train, X_test, y_train, y_test


def preprocess_bayes(df_full):
    """
    Preprocess all features.  Assumptions made:
    - missing values are put in category '101'
    - all categorical
    """
    feats = ['BALDIZZ_A',
             'PAIHDFC3M_A',
             'HRLOUDJOB_A',
             'ASEV_A',
             'SEX_A',
             'HYSTEV2_A',
             'PAIAMNT_A',
             'SHTCVD19NM1_A',
             'CFSEV_A']
    encoder = OneHotEncoder(drop='first', sparse_output=False).set_output(transform="pandas")
    y_name = 'LONGCOVD1_A'

    # drop rows without target
    df = df_full.dropna(subset=y_name)
    df = df[df[y_name] != 9]

    # Subset to columns
    bayes_df = df[feats]
    # Put missing pain into separate category, '101'
    bayes_df[bayes_df.isna()] = 101
    # Data type: int
    bayes_df = bayes_df.astype(int)
    
    cat_df = encoder.fit_transform(bayes_df)
    
    X = cat_df.copy()
    y = df.loc[:, y_name]
    y = (y == 1) * 1 # encode as 0 / 1 values
    
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=207)

    return X_train, X_test, y_train, y_test


def preprocess_chi(df_full):
    feats = ['MHRX_A','FDSCAT3_A','ANXLEVEL_A','PAIWKLM3M_A','EMERG12MTC_A','DISCRIM3_A','BALDIZZ_A','DEPLEVEL_A']

    encoder = OneHotEncoder(drop='first', sparse_output=False).set_output(transform="pandas")
    y_name = 'LONGCOVD1_A'
    
    # drop rows without target
    df = df_full.dropna(subset=y_name)
    df = df[df[y_name] != 9]
    
    # Subset to columns
    bayes_df = df[feats]
        
    # Put missing pain into separate category, '101'
    bayes_df[bayes_df.isna()] = 101
    # Data type: int
    bayes_df = bayes_df.astype(int)
    
    if np.any(bayes_df.apply(np.unique).apply(len) > 15):
        scaler = StandardScaler().set_output(transform="pandas")
    
        X = bayes_df.copy()
        
        # define continuous and categorical features
        continuous = X.columns[X.apply(np.unique).apply(len) > 15] 
        categorical = X.columns[~X.columns.isin(continuous)]
        
        # scale and encode
        cont_df =  scaler.fit_transform(X.loc[:, continuous])
        cat_df = encoder.fit_transform(X.loc[:,categorical])
        
        processed_df = pd.concat((cont_df, cat_df), axis=1)
    else:
        processed_df = encoder.fit_transform(bayes_df)
    
    X = processed_df.copy()
    y = df.loc[:, y_name]
    y = (y == 1) * 1 # encode as 0 / 1 values
    
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=207)
    return X_train, X_test, y_train, y_test


def preprocess_all_columns(df_full):
    """
    Preprocess all features.  Several assumptions are made:
    - null values of the target (long covid) are dropped
    - features with greater than 8000 missing values are dropped
    - values are imputed before the train test split, missing values are '101'
    - continuous values are arbitrarily defined
    """
    encoder = OneHotEncoder(drop='first', sparse_output=False).set_output(transform="pandas")
    scaler = MinMaxScaler().set_output(transform="pandas")
    imputer = SimpleImputer(strategy="constant", fill_value=101).set_output(transform="pandas")

    y_name = 'LONGCOVD1_A'
    
    # filter age
    new_df = df_full[df_full["AGEP_A"] <= 84].reset_index(drop=True)
    # filter ys
    # subset for features and drop null values
    df = new_df.dropna(subset=y_name)
    df = df[df[y_name] != 9]
    # drop null columns
    null_cols = df.columns[df.isna().all()].values
    low_counts = df.columns[df.isna().apply(sum) > 8000].values
    df_droppedcols = df.drop(columns=np.hstack((null_cols, low_counts, ['HHX'], y_name)))
    
    # impute values 
    imputed = imputer.fit_transform(df_droppedcols)

    # define continuous and categorical features
    continuous = df_droppedcols.columns[df_droppedcols.apply(np.unique).apply(len) > 15] 
    categorical = df_droppedcols.columns[~df_droppedcols.columns.isin(continuous)]

    # scale and encode
    cont_df =  scaler.fit_transform(imputed.loc[:, continuous])
    cat_df = encoder.fit_transform(imputed.loc[:,categorical])
    
    processed_df = pd.concat((cont_df, cat_df), axis=1)
    
    X = processed_df.copy()
    y = df.loc[:, y_name]
    y = (y == 1) * 1 # encode as 0 / 1 values
    
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=207)

    return X_train, X_test, y_train, y_test