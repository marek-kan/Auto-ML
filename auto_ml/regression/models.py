#from django.db import models
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.base import clone

import sys
sys.path.append('../')
from processing import *

class AutoRegression():
    def __init__(self, train, test):
        self.train = train
        self.test = test
        self.all_columns = list(train.columns)
        self.all_columns[-1] = 'y'
        self.train.columns = self.all_columns
        self.missing = train.isna().sum()
        self.drop_all = False # do you want to drop if missing any?
        self.filling_method = 'median'
        self.fill_categorical = 'most common'
        self.selection_treshold = 0.05
        self.plot_only = 20 # feature importencies plot
        self.splits = 3
        self.skf = KFold(n_splits=self.splits, shuffle=True)

    def create_data_types(self):
        for col in self.all_columns:
            try:
                if float(self.train[col].iloc[-3]):
                    self.train[col] = self.train[col].astype(np.float32)
            except:
                pass
        self.d_types = self.train.dtypes

    def pick_model(self):
        self.x = self.train[self.use_columns]
        try:
            self.x = pd.get_dummies(self.x)
        except:
            pass # if no categorical features
        self.final_columns = self.x.columns
        print(self.x.columns)
        self.scaler = StandardScaler()
        self.x = self.scaler.fit_transform(self.x)
        self.y = self.train['y']

        if len(np.unique(self.y))<50:
            print('Consider using classification, probably not continuos target variable!')

        # for picking the best model
        lr = Ridge(max_iter=1500)
        rf = RandomForestRegressor(n_estimators=500, max_depth=20, min_samples_leaf=3,
                                    max_features='auto', n_jobs=-1)
        svr = SVR(max_iter=-1)

        self.models = {'lr': lr, 'rf': rf, 'svr': svr}
        self.scores = {'lr': [], 'rf': [], 'svr': []}
        print('selecting model')
        for i, (train_index, test_index) in enumerate(self.skf.split(self.x, self.y)):
            x_tr, x_val = self.x[train_index], self.x[test_index]
            y_tr, y_val = self.y[train_index], self.y[test_index]
            if len(x_tr)>10000:
                print('reduced train size')
                y_tr.index, y_val.index = range(len(y_tr)), range(len(y_val))
                mask_train = np.random.choice(range(len(x_tr)),size=10000)
                x_tr, y_tr = x_tr[mask_train], y_tr[mask_train]
            for k, model in self.models.items():
                print('fold: ', i+1)
                print('model: ', k)
                model = clone(self.models[k])
                model.fit(x_tr, y_tr)
                p = model.predict(x_val)
        #        score = mean_squared_error(y_val, p)
                score = mean_absolute_error(y_val, p)
                self.scores[k] = self.scores[k] + [score]

        self.best_score = 9e10
        self.old_score = 9e10
        self.best_model = ''
        self.old_model = ''
        for k, l in self.scores.items():
            mean = np.mean(l)
            if mean < self.best_score:
                self.old_score = self.best_score
                self.old_model = self.best_model
                self.best_score = mean
                self.best_model = k
        print(self.best_model, self.best_score)

    def run(self):
        self.create_data_types()

        ### FILLING
        print('filling')
        if self.drop_all:
            self.train = self.train.dropna().reset_index()
            self.train.drop('index', axis=1, inplace=True)
        else:
            self.filler = HandleMissing(self.train, self.filling_method, self.fill_categorical)
            self.train = self.filler.fit_transform(self.train)

        ### SELECTING BEST
        print('selecting features')
        self.selector = SelectBest(self.d_types, self.train)
        self.use_columns = self.selector.select_features(thrash_numerical=self.selection_treshold,
                                                         thrash_categorical=self.selection_treshold)
        ### PICKING MODEL
        self.pick_model()

        model = clone(self.models[self.best_model])
        print('fitting the final model')
        model.fit(self.x, self.y)
        print(model)

        # get best params
        if self.best_model=='svr':
            if self.old_model == 'rf':
                temp = clone(self.models['rf'])
                temp.fit(x, y)
                weights = temp.feature_importances_
            else:
                temp = clone(self.models['lr'])
                temp.fit(x, y)
                weights = temp.coef_
        elif self.best_model=='rf':
            weights = model.feature_importances_
        else:
            weights = model.coef_

        try:
            weights = weights.reshape(weights.shape[1], -1)
        except:
            weights = weights.reshape(-1, 1)

        self.feature_imp_html = plot_feature_imp(weights, self.final_columns,
                                                 categories=None, to_plot=self.plot_only,
                                                 regression=True)

        if self.x.shape[1]!=1:
            print('Using PCA to visualize regression line!')
            pca = PCA(1)
            pca.fit(self.x)
            variance = pca.explained_variance_ratio_ # Percentage of variance explained by each of the selected components.
            print('Explained variance in data: ', str(sum(variance)*100)+'%')
            x_vis = pca.transform(self.x)
            model_vis = clone(self.models[self.best_model])
            model_vis.fit(x_vis, self.y)
        else:
            x_vis = self.x.copy()
            model_vis = model

        # create regression line
        self.reg_line_html = vis_regression_line(x_vis, self.y, model_vis)

        print('testing')
        #test = pd.read_csv(read_test).reset_index()
        #test.drop('index', axis=1, inplace=True)
        test_columns = list(self.test.columns)
        test_columns[-1] = 'y'
        self.test.columns = test_columns

        if self.drop_all:
            self.test = self.test.dropna()
        else:
            self.test = self.filler.transform(self.test) # in case some missing values in test sample

        y_test = self.test['y']
        x_test = self.test.drop('y', axis=1)
        x_test = x_test[self.use_columns]
        x_test = pd.get_dummies(x_test)

        x_test = dummy_handler(self.final_columns, list(x_test.columns), x_test)
        x_test = self.scaler.transform(x_test)

        pred = model.predict(x_test)

        self.mae_test = mean_absolute_error(y_test, pred)
        self.r2_test = r2_score(y_test, pred)

        print('Mean absolute error: ', self.mae_test)
        print('R^2 score: ', self.r2_test)
        print('Used Features: ', self.use_columns)
