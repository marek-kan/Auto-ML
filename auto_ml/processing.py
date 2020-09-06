# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 10:51:32 2020

@author: Marek
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, chi2
from io import BytesIO
import base64

class SelectBest():
    """Selects best features based on p-value threshold

    Parameters
    ----------
    d_types: Series, index - col name, value - data type
    df: DataFrame, data including Y
    """
    def __init__(self, d_types, df):
        self.types = d_types
        self.df = df
        self.columns = df.columns

    def select_numerical(self, threshold=0.05):
        """Selects best numerical features based on f-regression

        Parameters
        ----------
        threshold: float, trashold for feature significance

        Returns
        -------
        use_numerical: list, list of selected features
        """
        cols = self.types[self.types!=np.object]
        if len(cols[cols.index!='y'])==0:
            return [] # no numerical features
        df_temp = self.df[cols.index]
        x_temp = df_temp.drop('y', axis=1)
        y_temp = df_temp['y']

        selector = SelectKBest(f_regression, k='all')
        selector.fit(x_temp, y_temp)
        res = selector.pvalues_
        result = pd.DataFrame()
        result['columns'] = x_temp.columns
        result['p'] = res
        self.result_numerical = result

        use_numerical = list(result[result['p']<threshold]['columns'])
        return use_numerical

    def select_categorical(self, threshold=0.05):
        """Selects best numerical features based on chi^2

        Parameters
        ----------
        threshold: float, trashold for feature significance

        Returns
        -------
        use_cat: list, list of selected features
        """
        cols = self.types[self.types==np.object]

        df_temp = self.df[cols.index]

        x_temp = df_temp.copy()
        try:
            x_temp = pd.get_dummies(x_temp)
        except:
            return [] # if no categorical features
        y_temp = self.df['y']

        selector = SelectKBest(chi2, k='all')
        selector.fit(x_temp, y_temp)
        res = selector.pvalues_
        result = pd.DataFrame()
        result['columns'] = x_temp.columns
        result['p'] = res
        self.result_categorical = result

        use_cat = list(result[result['p']<threshold]['columns'])
        use_cat = list(set([x.split('_')[0] for x in use_cat]))
        return use_cat

    def select_features(self, thrash_numerical=0.05, thrash_categorical=0.05):
        numerical = self.select_numerical(thrash_numerical)
        categorical = self.select_categorical(thrash_categorical)
        temp = numerical + categorical
        while len(temp) == 0:
            print('No predictive features at chosen thrashold, increasing trasholds')
            thrash_numerical, thrash_categorical = thrash_numerical+0.05, thrash_categorical+0.05
            print('Current thrasholds numerical/categorical: ', thrash_numerical, thrash_categorical)
            numerical = self.select_numerical(thrash_numerical)
            categorical = self.select_categorical(thrash_categorical)
            temp = numerical + categorical
        # takes care of column names with multiple "_"
        result = []
        for feature in temp:
            for col in self.columns:
                if col.startswith(feature):
                    result.append(col)
        result = list(set(result)) # avoid duplicates
        return result


class HandleMissing():
    """Handles missing data

    Parameters
    ----------
    filling_num: str, how to handle missing numerical values, mean/median
    filling_cat: str, how to handle missing categorical values, most common/drop/na category
    """
    def __init__(self, data, filling_num, filling_cat):
        self.filling_num = filling_num
        self.filling_cat = filling_cat
        self.d_types = data.dtypes
        self.filling_dict = None

    def replace_missing(self, df, missing, fit=True):
        """Replaces missing values, creates dict of filled values

        Parameters
        ----------
        df: DataFrame, train/test sample
        missing: Series, index - col names, value - # missing values
        fit: True for train set, False for test set

        Returns
        -------
        df: DataFrame
        """
        if fit:
            for col in missing.index:
                if missing[col] == 0 and self.d_types[col]!=np.object:
                    continue
                elif missing[col] == 0 and self.d_types[col]==np.object:
                    if self.filling_cat == 'most common':
                        counts = df[col].value_counts()
                        self.filling_dict.update({col: counts.index[0]})
                    elif self.filling_cat == 'encode':
                        self.filling_dict.update({col: 'filled_nan_value'})
                else:
                    if self.d_types[col].name.startswith('int') or self.d_types[col].name.startswith('float'):
                        df[col].fillna(self.filling_dict[col], inplace=True)
                    else:
                        if self.filling_cat == 'most common':
                            counts = df[col].value_counts()
                            self.filling_dict.update({col: counts.index[0]})
                            df[col].fillna(counts.index[0], inplace=True) # most common variable
                        elif self.filling_cat == 'drop':
                            df.dropna(subset=[col], inplace=True)
                        elif self.filling_cat == 'encode':
                            self.filling_dict.update({col: 'filled_nan_value'})
                            df[col].fillna('filled_nan_value', inplace=True)
        else:
            for col in missing.index:
                if missing[col] == 0:
                    continue
                else:
                    if self.d_types[col].name.startswith('int') or self.d_types[col].name.startswith('float'):
                        df[col].fillna(self.filling_dict[col], inplace=True)
                    else:
                        if self.filling_cat == 'drop':
                            df.dropna(subset=[col], inplace=True)
                        else:
                            df[col].fillna(self.filling_dict[col], inplace=True)
        return df

    def fit_transform(self, df):
        missing = df.isna().sum()
        if self.filling_num=='mean':
            self.filling_dict = df.mean().to_dict()
        elif self.filling_num=='median':
            self.filling_dict = df.median().to_dict()

        df = self.replace_missing(df, missing)
        return df

    def transform(self, df):
        if self.filling_dict == None:
            return 'You need fit_transform first!'

        missing = df.isna().sum()
        df = self.replace_missing(df, missing, fit=False)
        return df

def plot_feature_imp(weights, columns, categories=None, to_plot=10, regression=False):
    """Plots best n feature importancies (weights)

    Parameters
    ----------
    weights: array, weights from estimator
    categories: list, target categories (for binary cassification task len = 2)
    columns: list, names of features
    to_plot: int, how many features to plot

    Returns
    -------
    HTML img tag
    """
    n_weights = len(weights)
    if n_weights < to_plot:
        to_plot = n_weights
    if regression:
        best_predictors = pd.DataFrame()
        best_predictors['feature'] = columns
        best_predictors['weight'] = weights
        best_predictors['abs_weight'] = abs(best_predictors['weight'])
        best_predictors.sort_values('abs_weight', ascending=False, inplace=True)

        plt.figure() # figsize=(16, 8)
        plot_df = best_predictors.iloc[:to_plot, :].sort_values('weight', ascending=False)
        plt.barh(range(to_plot), plot_df['weight'])
        plt.yticks(range(to_plot), plot_df['feature'])
        plt.title(f'Top {to_plot} features')
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png')
        data = base64.b64encode(buf.getvalue()).decode('utf8')
    else:
        if weights.shape[1]==1:
            best_predictors = pd.DataFrame()
            best_predictors['feature'] = columns
            best_predictors['weight'] = weights
            best_predictors['abs_weight'] = abs(best_predictors['weight'])
            best_predictors.sort_values('abs_weight', ascending=False, inplace=True)

            plt.figure(figsize=(16,8))
            plot_df = best_predictors.iloc[:to_plot, :].sort_values('weight', ascending=False)
            plt.barh(range(to_plot), plot_df['weight'])
            plt.yticks(range(to_plot), plot_df['feature'])
            plt.title(f'Top {to_plot} features')
            buf = BytesIO()
            plt.savefig(buf, format='png')
            data = base64.b64encode(buf.getvalue()).decode('utf8')
        else:
            fig, axs = plt.subplots(len(categories), 1, figsize=(8, 20), facecolor='w', edgecolor='k')
            fig.subplots_adjust(hspace = .5)
            axs = axs.ravel()
            for i, category in enumerate(categories):
                best_predictors = pd.DataFrame()
                best_predictors['feature'] = columns
                best_predictors[f'weight_{category}'] = weights[:, i]
                best_predictors['abs_weight'] = abs(best_predictors[f'weight_{category}'])
                best_predictors.sort_values('abs_weight', ascending=False, inplace=True)

                plot_df = best_predictors.iloc[:to_plot, :].sort_values(f'weight_{category}', ascending=False)
                axs[i].barh(range(to_plot), plot_df[f'weight_{category}'])
                axs[i].set_yticks(range(to_plot))
                axs[i].set_yticklabels(plot_df['feature'])
                axs[i].set_title(f'Top {to_plot} features for {category} category')
            buf = BytesIO()
            fig.savefig(buf, format='png')
            data = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close()
    return f"<img src=\'data:image/png;base64,{data}\'>"


def dummy_handler(train_cols, test_cols, data):
    """Validates testing data

    Parameters
    ----------
    train_cols: list, all training columns, after processing
    test_cols: list, all testing columns, after processing
    data: test dataset

    Returns
    -------
    data: Data Frame
    """
    # Get missing columns in the training test
    missing_cols = set(train_cols) - set(test_cols)
    # Add a missing column in test set with default value equal to 0
    for c in missing_cols:
        data[c] = 0
    # Ensure the order of column in the test set is in the same order than in train set
    data = data[train_cols]
    return data

def vis_regression_line(x, y, model):
    """Plots regression line

    Parameters
    ----------
    x: array, shape (n, 1), feature
    y: array, shape (n, 1), target variable
    model: estimator

    Returns
    -------
    HTML img tag
    """
    dummy_x = np.array([x.min(), np.median(x), x.max()]).reshape(-1, 1)
    plt.scatter(x, y)
    plt.plot(dummy_x, model.predict(dummy_x), label='Regression line', color='red',
             linewidth=3)
    plt.legend()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    data = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close()
    return f"<img src=\'data:image/png;base64,{data}\'>"

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(clf, xx, yy, X, Y,**params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    #ax: matplotlib axes object, replaced by plt
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, **params)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('Decision boundary')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    data = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close()
    return f"<img src=\'data:image/png;base64,{data}\'>"
