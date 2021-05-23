'''
    This file is to prepare data for a machine learning model, so it includes:
    - data pre-processing
    - feature engineering (creation, deletion, extraction, scaling)
'''
# Let's import libraries
import pandas as pd # to handle dataframes
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
import seaborn as sns
import datetime # manipulating date formats
import warnings
warnings.filterwarnings("ignore")


# Let's build a function to differentiate the categorical and numerical columns of the dataset
def __cat_or_num__(df):
    '''
    Function that clasifies the kind of columns (categorical or numerical) that exist in a dataset
    Input:
            df: dataset where we want to clasify columns
    Output:
            num_cols,cat:cols: it returns a dictionary with the category of every column

    '''
    cat_cols = []
    num_cols = []

    num_cols = [i for i in df.columns if df.dtypes[i] != 'object']
    cat_cols = [i for i in df.columns if df.dtypes[i] == 'object']

    return cat_cols, num_cols


# get information about the cols and features
def show_cols_info(df):
    '''
    Function that shows the summary of the information of the columns of the datasets
    Input:
            df: dataset we want to show information
    Output:
            none: it prints a summary and the main values of every column

    '''

    for column in df.columns:
        # show a summary of the important information of the column
        df.describe([column]).show()
        # show the different values of the field
        df.select([column]).distinct().show()


def __load_data__(type):
    '''
    Function that loads data into Pandas dataframes
    Input: type of dataset (train/test) we want to load
    Output: df dataset is loaded into memory
    '''
    if type == 'train':
        df = pd.read_csv(os.path.join('../../Rossmann/data/', 'train.csv'), low_memory=False,parse_dates = True)
    elif type == 'test':
        df = pd.read_csv(os.path.join('../../Rossmann/data/', 'test.csv'), low_memory=False,parse_dates = True)
    else:
        raise ValueError('dataset type must be "train" or "test".')

    # merge dataframes to get store information in both training and testing sets
    df_store = pd.read_csv(os.path.join('../../Rossmann/data/', 'store.csv'), low_memory=False,parse_dates = True)
    df = df.merge(df_store, on='Store')

    return df

def __handle_categorical__(df):
    '''
    Function to select categorical variables and encode them
    Input: df: Pandas dataframe containing categorical variables
    Output: modified StateHoliday, StoreType and Assortment columns
    '''

    # define a dict where the keys are the element to encode, and the values are the targets
    mappings = {'0': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4}

    # now we can encode the categorical features
    df['StateHoliday'].replace(mappings, inplace=True)
    df['StoreType'].replace(mappings, inplace=True)
    df['Assortment'].replace(mappings, inplace=True)

    return df

def __preprocess_features__(df):
    '''
    Function to add information to the date such as month, day, etc ...
    Input:  df: Pandas dataframe
    Output: df: Pandas dataframe

    '''

    def __get_months_since_competition__(year, competition_open_since_year, month, competition_open_since_month):
        if competition_open_since_year > 0 and competition_open_since_month > 0:
            return 12 * (year - competition_open_since_year) + (month - competition_open_since_month)
        else:
            return 0

    def __get_month_since_promo__(year, promo2_since_year, week_of_year, promo2_since_week):
        if promo2_since_week > 0 and promo2_since_year > 0:
            return 12 * (year - promo2_since_year) + (week_of_year - promo2_since_week) / 4.

    # Let's deal with null values...

    # Let's impute with 0 when the competitionDistance,CompetitionOpenSinceMonth
    # and CompetitionOpenSinceYear is null
    df['CompetitionDistance'].fillna(0, inplace=True)
    df['CompetitionOpenSinceMonth'].fillna(0,inplace=True)
    df['CompetitionOpenSinceYear'].fillna(0,inplace=True)

    # We can guess that if there is no Promo2, the Promo2SinceWeek is null
    # Let's impute with 0
    df['Promo2SinceWeek'].fillna(0, inplace=True)
    df['Promo2SinceYear'].fillna(0, inplace=True)
    df['PromoInterval'].fillna(0, inplace=True)

    # Now let's change some formats
    # Let's format the date column
    df['Date'] = pd.to_datetime(df['Date'])
    #df.Date = df.Date.apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
    
    # Let's augment with some more information
    # Let's augment with some data

    df['Date'] = pd.to_datetime(df['Date'])
    df['Day'] = df['Date'].apply(lambda x: x.day)
    df['Month'] = df['Date'].apply(lambda x: x.month)
    df['Year'] = df['Date'].apply(lambda x: x.year)
    df['DayOfWeek'] = df['Date'].apply(lambda x: x.dayofweek)
    df['Week'] = df['Date'].apply(lambda x: x.weekofyear)

    df['is_quarter_start'] = df['Date'].dt.is_quarter_start
    df['is_quarter_end'] = df['Date'].dt.is_quarter_end
    df['is_month_start'] = df['Date'].dt.is_month_start
    df['is_month_end'] = df['Date'].dt.is_month_end
    df['is_year_start'] = df['Date'].dt.is_year_start
    df['is_year_end'] = df['Date'].dt.is_year_end

    df.is_quarter_start=df.is_quarter_start.apply(lambda x: 0 if x==False else 1)
    df.is_quarter_end=df.is_quarter_end.apply(lambda x: 0 if x==False else 1)
    df.is_month_start=df.is_month_start.apply(lambda x: 0 if x==False else 1)
    df.is_month_end=df.is_month_end.apply(lambda x: 0 if x==False else 1)
    df.is_year_start=df.is_year_start.apply(lambda x: 0 if x==False else 1)
    df.is_year_end=df.is_year_end.apply(lambda x: 0 if x==False else 1)

    df.drop(['Date'], inplace=True, axis=1)

    # number of months since a competition store has opened
    df['MonthsSinceCompetition'] = df.apply(
        lambda row: __get_months_since_competition__(row['Year'], row['CompetitionOpenSinceYear'], row['Month'],
                                                 row['CompetitionOpenSinceMonth']), axis=1)
    df.drop(['CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth'], inplace=True, axis=1)

    # number of months since a promotion has started
    df['MonthsSincePromo'] = df.apply(
        lambda row: __get_month_since_promo__(row['Year'], row['Promo2SinceYear'], row['Week'],
                                              row['Promo2SinceWeek']), axis=1)
    df['MonthsSincePromo'] = df['MonthsSincePromo'].apply(lambda x: x if x > 0 else 0)
    df.drop(['Promo2SinceYear', 'Promo2SinceWeek', 'PromoInterval', 'Store'], inplace=True, axis=1)

    return df
    
def build_dataset(type):
    '''
    Function to build the dataset
    Input:  df: Pandas dataframe
    Output: df: Pandas dataframe
    '''
    df = __load_data__(type)
    df = __handle_categorical__(df)
    df = __preprocess_features__(df)

    return df

def save_dataset(df, filename):
    '''
    Function to save a dataset to a csv file
    Input: df: dataset to save
    Output: filename: file to save the dataset
    '''

    path = os.path.join('', filename)
    df.to_csv(path, index=False)
    print('Saving dataset to:', filename)

class Preprocessor:
    def __init__(self):
        self.data_stats = {}
        pass

    def fit(self, data):
        '''
        We suppose that the train and test data come from the same distribution, so we need to fit the preprocessing
        on the train data and later transform the test data with respect to the distribution of the train data.
        This method saves train data statistics that will be needed to fill missing values

        Input data: data set from which statistics are saved
        '''

        print('Fitting data...')
        # save the mean of this column for transform
        self.data_stats['MonthsSinceCompetitionMean'] = math.floor(data['MonthsSinceCompetition'].mean())
        self.data_stats['timestamp'] = datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")

        # save data_stats to pickle file.
        # this file will be necessary when preprocessing test data
        print('Saving data stats to pickle file...')
        data_stats_file = open('data_stats.pkl', 'wb')
        pickle.dump(self.data_stats, data_stats_file)
        data_stats_file.close()
        print('Fitting data done.')

    def transform(self, data):
        """
        Fills missing values with means saved from training data and scales target
        :param data: dataset to transform
        """

        # if object has not been fit prior to transform call, load data stats from pickle file
        if not self.data_stats:
            data_stats_file = open('data_stats.pkl', 'rb')
            self.data_stats = pickle.load(data_stats_file)
            data_stats_file.close()

        print('Transforming data with training data statistics saved on:', self.data_stats['timestamp'])

        # fill missing values with mean
        data['MonthsSinceCompetition'] = data['MonthsSinceCompetition'].fillna(
            self.data_stats['MonthsSinceCompetitionMean'])

        data['CompetitionDistance'].fillna(0, inplace=True)

        # Fill with 0 promo2 variables
        data['MonthsSincePromo'].fillna(0, inplace=True)

        data['Open'].fillna(0, inplace=True)

        # scale target ('Sales')
        if 'Sales' in data.columns.tolist():
            data['Sales'] = data['Sales'].apply(lambda x: np.log(x) if x > 0 else 0)

        print('Transforming data done!')

        return data

def __get_most_important_features__(X_train, y_train, n_features):
    '''
    Perform feature selection using XGBoost algorithm
    '''
    model = XGBClassifier()
    model.fit(X_train, y_train)
    sorted_idx = np.argsort(model.feature_importances_)[::-1]
    sorted_idx = sorted_idx[:n_features]
    return X_train.columns[sorted_idx]

# unit test
if __name__ == "__main__":
    data = build_dataset('train')
    preprocessor = Preprocessor()
    preprocessor.fit(data)
    data = preprocessor.transform(data)