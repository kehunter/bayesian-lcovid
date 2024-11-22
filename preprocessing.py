import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from zipfile import ZipFile
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

## USE ALL FEATURES

def preprocess_all_columns():
    """
    Preprocess all features.  Several assumptions are made:
    - null values of the target (long covid) are dropped
    - features with greater than 8000 missing values are dropped
    - values are imputed before the train test split
    - continuous values are arbitrarily defined
    """
    # read in data
    zip_file = ZipFile("data/adult23csv.zip")
    
    df_full = pd.read_csv(zip_file.open('adult23.csv'))
    y_name = 'LONGCOVD1_A'
    
    encoder = OneHotEncoder(drop='first', sparse_output=False).set_output(transform="pandas")
    scaler = StandardScaler().set_output(transform="pandas")
    imputer = SimpleImputer(strategy='most_frequent').set_output(transform="pandas")
    
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
    ## END PREPROCESSING
    
    X = processed_df.copy()
    y = df.loc[:, y_name]
    y = (y == 1) * 1 # encode as 0 / 1 values
    
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=207)

    return X_train, X_test, y_train, y_test




def preprocess_after_bayes():
    """
    Preprocess all features.  Assumptions made:
    - null values of the target (long covid) are dropped
    - features selected from bayes with any null values are dropped
    """

    feats = ['PRDEDUC1_A',
     'CITZNSTP_A',
     'VISIONDF_A',
     'BPLAST_A',
     'PCNTADWKP1_A',
     'PAIAFFM3M_A',
     'PHQ43_A',
     'COLORECTEV_A',
     'HEARINGDF_A',
     'SOCWRKLIM_A',
     'ARTHEV_A',
     'VIMREAD_A',
     'ANXFREQ_A',
     'MRIHAD_A',
     'HOUYRSLIV_A',
     'COLOGUARD1_A',
     'CFSEV_A',
     'UPPRAISE_A',
     'CBALHDINJ_A',
     'VIGIL3_A',
     'DISCRIM5_A',
     'DISCRIM1_A',
     'INCWRKO_A',
     'PRDNCOV1_A',
     'NUMCAN_A',
     'FSNAP12M_A',
     'SAPARENTSC_A',
     'CROHNSEV_A',
     'CHLEV_A',
     'VIMDREV_A',
     'PAITOOTH3M_A',
     'COMDIFF_A',
     'PAYBLL12M_A',
     'PRRXCOV1_A',
     'DIFF_A',
     'STREV_A',
     'DISCRIM4_A',
     'PARSTAT_A',
     'EXCHPR1_A',
     'PLNWRKR1_A',
     'FDSBALANCE_A',
     'SHTCVD19NM1_A',
     'CTCOLEV1_A',
     'COGMEMDFF_A',
     'EMPHEALINS_A',
     'PAIHDFC3M_A',
     'MEDICAID_A',
     'ANXEV_A',
     'PCNTADWFP1_A',
     'URGNT12MTC_A',
     'SHTFLUM_A',
     'EMERG12MTC_A',
     'DEPFREQ_A',
     'FDSRUNOUT_A',
     'VIMMDEV_A',
     'AVAIL_A',
     'FDSLAST_A',
     'VIGIL2_A',
     'HOUSECOST_A',
     'DEPMED_A',
     'DEPLEVEL_A',
     'DEPEV_A',
     'ACCSSINT_A',
     'EDUCP_A',
     'INCSSISSDI_A',
     'DISCRIM3_A',
     'MEDICARE_A',
     'VIRAPP12M_A',
     'MEDRXTRT_A',
     'MAMEV_A',
     'VIGIL4_A',
     'MAXEDUCP_A']

    
    # read in data
    zip_file = ZipFile("data/adult23csv.zip")
    
    df_full = pd.read_csv(zip_file.open('adult23.csv'))
    y_name = 'LONGCOVD1_A'
    
    encoder = OneHotEncoder(drop='first', sparse_output=False).set_output(transform="pandas")
    
    df = df_full.dropna(subset=y_name)
    df = df[df[y_name] != 9]
    bayes_df = df[feats].loc[:,~df[feats].isna().any()]
    bayes_df.apply(lambda x:x.unique())
    bayes_df
    
    cat_df = encoder.fit_transform(bayes_df)
    
    X = cat_df.copy()
    y = df.loc[:, y_name]
    y = (y == 1) * 1 # encode as 0 / 1 values
    
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=207)

    return X_train, X_test, y_train, y_test

    
def preprocess_data():
    # read in data
    zip_file = ZipFile("data/adult23csv.zip")
    
    df_full = pd.read_csv(zip_file.open('adult23.csv'))
    y_name = 'LONGCOVD1_A'
    
    ## FEATURE INFORMATION
    
    demographic_info = {
        "AGEP_A": "Age of Sample Adult (top-coded)",
        "SEX_A": "Sex of Sample Adult",
        "HISP_A": "Hispanic ethnicity of Sample Adult",
        "HISPALLP_A": "Single and multiple race groups with Hispanic origin",
        "RACEALLP_A": "Single and multiple race groups",
        "MARITAL_A": "Sample adult's current marital status",
        "SPOUSLIV_A": "Sample adult's spouse lives here",
        "EDUCP_A": "Educational level of sample adult",
        "PCNT18UPTC": "Top-coded count of persons 18 or older in the household",
        "OVER65FLG_A": "Indicator for at least 1 person aged 65+ in Sample Adult family",
        "MLTFAMFLG_A": "Indicator for multifamily households",
        "PCNTKIDS_A": "Number of children in Sample Adult family, top-coded 3+",
        "PARSTAT_A": "Parental Status of sample adult",
        "CEVOLUN1_A": "Volunteer for organization or association",
        "CEVOLUN2_A": "Other volunteer activities",
        "FWIC12M_A": "Receive WIC benefits, past 12 months",
        "INCWRKO_A": "Income from wages"
    }
    
    chronic_info = {
        "EVERCOVD_A": "Ever had COVID-19",
        "LONGCOVD1_A": "Had COVID-19 symptoms for 3 or more months",
        # "SYMPNOW1_A": "Currently has COVID-19 symptoms",
        # "LCVDACT_A": "COVID-19 impacts activities",
        "DEPEV_A": "Ever had depression",
        "ANXEV_A": "Ever had anxiety disorder",
        "COPDEV_A": "Ever been told you had COPD, emphysema, or chronic bronchitis?",
        "ARTHEV_A": "Ever had arthritis",
        "DEMENEV_A": "Ever had dementia",
        "DIBTYPE_A": "Diabetes type",
        "HEPEV_A": "Ever had hepatitis",
        "CROHNSEV_A": "Ever had Crohn's disease",
        "ULCCOLEV_A": "Ever had ulcerative colitis",
        "CFSEV_A": "Ever had Chronic Fatigue Syndrome",
        "HLTHCOND_A": "Weakened immune system due to health condition",
        "MEDRXTRT_A": "Weakened immune system due to prescriptions",
        "PAIAMNT_A": "How much pain last time",
        "PAIWKLM3M_A": "How often pain limits life or work",
        "DENPREV_A": "Time since last dental exam or cleaning",
        "DENDL12M_A": "Delayed dental care due to cost, past 12 months",
        "SHTCVD191_A": "COVID-19 vaccination",
        "SHTCVD19NM1_A": "Number of COVID-19 vaccinations",
        "SHOTTYPE2_A": "Brand of first COVID-19 shot",
        "CVDVAC1M1_A": "Month of most recent COVID-19 vaccination",
        "CVDVAC1Y1_A": "Year of most recent COVID-19 vaccination",
        "SHTPNUEV_A": "Ever had pneumonia shot",
        "SHTPNEUNB_A": "Number of pneumonia shots",
        "SHTSHINGL1_A": "Ever had a shingles vaccination"
    }
    
    # list of all columns to select
    feat_names = list(chronic_info.keys()) + list(demographic_info.keys())
    
    
    ## TRAIN TEST SPLIT FROM FULL FEATURES
    
    # subset for features and drop null values
    df = df_full.loc[:, feat_names].dropna(subset=y_name)
    df = df[df[y_name] != 9]
    
    # # FIX ATTRIBUTES
    # df["AGEP_A"] = (df["AGEP_A"] == 1) * 1  # 1 is male 
    # df["SHTCVD191_A"] = (df["SHTCVD191_A"] == 1) * 1  # 1 is yes covid shot
    
    # split X and y
    X = df.loc[:, df.columns != y_name]
    y = df.loc[:, y_name]
    y = (y == 1) * 1 # encode as 0 / 1 values
    
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=207)
    
    return X_train, X_test, y_train, y_test
    


def preprocess_after_second_bayes(df_full):
    """
    Preprocess all features.  Assumptions made:
    - missing values for hysterectomy are marked 'no hysterectomy'
    - other missing values are put in category '0'
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
    # Assume missing hyterectomy values are 'no'
    bayes_df['HYSTEV2_A'].loc[bayes_df['HYSTEV2_A'].isna()] = 2
    # Put missing pain into separate category, '0'
    bayes_df[bayes_df.isna()] = 0
    # Data type: int
    bayes_df = bayes_df.astype(int)
    
    
    cat_df = encoder.fit_transform(bayes_df)
    
    X = cat_df.copy()
    y = df.loc[:, y_name]
    y = (y == 1) * 1 # encode as 0 / 1 values
    
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=207)

    return X_train, X_test, y_train, y_test