import os
import urllib
from IPython.display import display
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_profiling import ProfileReport
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
import phik
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class dataLoader:
    def __init__(self):
        self._ROOT_DIRECTORY = "./data/"
        self._TRAIN_LOCAL_PATH = "./data/train.csv"
        self._TEST_LOCAL_PATH = "./data/test.csv"
        self._SUBMISSION_LOCAL_PATH = "./results/submission.csv"

    def batch_loader(self, path, is_csv, file_sep = ","):
        '''
        Loads the dataset at the given path, in various formats
        '''
        if not is_csv:
            return pd.read_parquet(path)
        else:
            return pd.read_csv(path, sep = file_sep)  
    
    def cloud_loader(self, url, name, is_csv, file_sep = ","):
        '''
        Fetches a file from an URL and loads it in various formats
        '''
        file = urllib.request.urlretrieve(url, self._ROOT_DIRECTORY + name)
        if not is_csv:
            return pd.read_parquet(file)
        else:
            return pd.read_csv(file, sep = file_sep)  
    
class dataSplitter:
    def __init__(self, df):
        self._df = df

    def train_splitter(self, target="", size=0.2, seed=42, is_Stratified= False):
        '''
        Splits your dataset randomly or stratified according to a target feature
        '''
        if is_Stratified:
            return train_test_split(self._df, self._df[target], test_size = size, random_state = seed, stratify = self._df[target])
        else:
            return train_test_split(self._df, test_size = size, random_state = seed)
            
    

class dataExplorer:
    def __init__(self, df, categorical, numerical):
        self._SMALL_SIZE = 24
        self._MEDIUM_SIZE = 32
        self._BIGGER_SIZE = 48
        self._df = df
        self._categorical = categorical
        self._numerical = numerical

    def basic_explorer(self, target):
        '''
        Returns an overview, basic information and statistics of your dataset
        '''
        display(self._df.head(5))
        display(self._df.info())    
        display(self._df.describe())         
        display(self._df[target].value_counts())
    
    def profile_explorer(self):
        '''
        Returns the pandas-profiling ProfileReport
        '''
        profile = ProfileReport(self._df)
        display(profile)
    
    def outlier_explorer(self, features=[]):
        '''
        Plots a Boxplot to analyze prossible Outliers 
        '''
        plt.rc('font', size = self._SMALL_SIZE)         
        plt.rc('axes', titlesize = self._SMALL_SIZE)     
        plt.rc('axes', labelsize = self._MEDIUM_SIZE)    
        plt.rc('xtick', labelsize = self._SMALL_SIZE)    
        plt.rc('ytick', labelsize = self._SMALL_SIZE)    
        plt.rc('legend', fontsize = self._SMALL_SIZE)    
        plt.rc('figure', titlesize = self._BIGGER_SIZE) 
        fig = plt.figure(figsize =(self._SMALL_SIZE, self._SMALL_SIZE))
        ax = fig.add_subplot()
        ax.boxplot([self._df[features]], labels =[features])
        ax.set_title('Outliers')
        ax.set_xlabel('Feature')
        ax.set_ylabel('Value')
        display(plt.show());

    
    def scatter_explorer(self, numerical, size):
        '''
        Plots an scatter matrix of the numerical fetures of a given dataset
        '''
        display(scatter_matrix(self._df[numerical], figsize = (size, size)));


    def correlation_explorer(self, target, numerical, is_phik, size):
        '''
        Returns a Pearsons R correlation matrix or an Phik matrix for both categorical and numerical features
        '''
        if not is_phik:
            corr_matrix = self._df.corr(numerical)
        else:
            corr_matrix = self._df.phik_matrix()
        fig, ax = plt.subplots(figsize=(size, size))         
        display(sns.heatmap(corr_matrix, annot=True, linewidths=.5, ax=ax))
        print(corr_matrix[target].sort_values(ascending = False))



class dataProcessor:
    def __init__(self, df, categorical, numerical) :
        self._df = df
        self._categorical = categorical
        self._numerical = numerical

    
    def imputer_processor(self, is_drop, is_mean = False, is_median = False, is_constant = False, constant = 0):
        '''
        Returns an imputed missing values dataset
        '''
        missing = self._df.isnull().value_counts()
        print(missing)
        if is_drop:
            return self._df[missing].dropna()
        elif is_mean:
            return self._df[missing].fillna((self._df[missing].mean()), inplace = True)
        elif is_median:
            return self._df[missing].fillna((self._df[missing].median()), inplace = True)
        elif is_constant:
            return self._df[missing].fillna(constant, inplace = True)
        else:
            return self._df.drop([missing], axis = 1)

    
    def outlier_processor(self, std_threshold = 3):
        '''
        Drops the outliers of a dataset
        '''
        return self._df[(np.abs(stats.zscore(self._df)) < std_threshold).all(axis = 1)]


    def encoder_processor(self, is_OHE = True):
        '''
        Applies OneHotEncoding to a given dataset
        '''
        if is_OHE:
            return pd.get_dummies(self._df[self._categorical])
        else:
            return self._df

    def scaler_processor(self, is_Standard = True, is_MinMax = False, is_Robust = False):
        ''' 
        Scales the dataset with Standard/MinMax/RobustScaler
        '''
        if is_Standard:
            scaler = StandardScaler()
        elif is_MinMax:
            scaler = MinMaxScaler()
        elif is_Robust:
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        self._df[self._numerical] = scaler.fit_transform(self._df[self._numerical])
        return self._df



