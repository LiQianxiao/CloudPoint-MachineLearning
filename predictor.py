'''
File: predictor.py
File Created: Sunday, 4th November 2018 7:34:47 pm
Author: Qianxiao Li (liqix@ihpc.a-star.edu.sg)
        Gonzalez Oyarce Anibal Lautaro (anibal-gonza@ihpc.a-star.edu.sg)
-----
License: MIT License
'''

import tensorflow as tf
import numpy as np
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBRegressor


class Regressor(object):
    def __init__(self, options):
        raise NotImplementedError

    def predict(self, input):
        raise NotImplementedError

    def fit(self, input, target, options):
        raise NotImplementedError


class NeuralNetRegressor(Regressor):

    def __init__(self, options):
        self.options = options
        self.input = tf.placeholder(
            tf.float32, [None, ] + options['input_shape'], 'input')
        self.target = tf.placeholder(
            tf.float32, [None, ] + options['target_shape'], 'target')
        self.name = options['name']
        self.session = options['session']

        with tf.variable_scope(
                self.name, reuse=False,
                initializer=tf.truncated_normal_initializer()):
            self.output = self._compute_output(self.input)
            self._add_loss_and_training_ops()
            self.variables = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
            self._initialize()

    def _initialize(self):
        self.session.run(tf.variables_initializer(self.variables))

    def _mlp(self, input, n_layers, layer_widths):
        hidden = input
        for n in range(n_layers):
            width = layer_widths[n]
            hidden = tf.layers.dense(
                hidden, width, activation=tf.nn.sigmoid,
                name='hidden_{}'.format(n))
        output = tf.layers.dense(
            hidden, self.options['target_shape'][-1], name='output')
        return output

    def _compute_output(self, input):
        return self._mlp(
            input,
            self.options['n_layers'],
            self.options['layer_widths'])

    def _add_loss_and_training_ops(self):
        self.loss = tf.losses.mean_squared_error(
            labels=self.target, predictions=self.output)
        opt = tf.train.AdamOptimizer()
        self.train_step = opt.minimize(self.loss)

    def _evaluate_loss(self, input, target):
        feed_dict = {self.input: input, self.target: target}
        return self.session.run(self.loss, feed_dict)

    def fit(self, input, target,
            options={'monitor': False, 'reinitialize': True,
                     'n_iter': 5000}):
        if options['reinitialize']:
            self._initialize()
        n_iter = options['n_iter']
        train_dict = {self.input: input, self.target: target}
        if options['monitor']:
            validation_input = options['validation_input']
            validation_target = options['validation_target']
            train_losses = [
                self._evaluate_loss(input, target)]
            validation_losses = [
                self._evaluate_loss(validation_input, validation_target)]

        for n in range(n_iter):
            self.session.run(self.train_step, train_dict)
            if options['monitor']:
                train_losses.append(
                    self._evaluate_loss(input, target))
                validation_losses.append(
                    self._evaluate_loss(validation_input, validation_target))
                if n % 100 == 0:
                    print('Iter: ', n,
                          'Train Loss: ', train_losses[-1],
                          'validation Loss: ', validation_losses[-1])

        if options['monitor']:
            return {'train_loss': train_losses,
                    'validation_loss': validation_losses}

    def predict(self, input):
        feed_dict = {self.input: input}
        return self.session.run(self.output, feed_dict)


class EnsembleRegressor(object):

    def __init__(self, regressors, scaler):
        self.regressors = list(regressors)
        self.scaler = scaler

    def predict(self, input):
        return np.asarray([r.predict(input) for r in self.regressors])

    def predict_mean(self, input):
        predictions = self.predict(input)
        return np.mean(predictions, axis=0)

    def predict_covariance(self, input):
        predictions = self.predict(input)
        mean = predictions.mean(axis=0, keepdims=True)
        residual = predictions - mean
        residual = residual.transpose([1, 0, 2])
        covariance = (residual.transpose(0, 2, 1) @ residual)
        covariance /= len(self.regressors)
        return covariance

    def fit(self, input, target, list_of_options):
        for regressor, options in zip(self.regressors, list_of_options):
            print('Fitting Regressor: '+regressor.name)
            regressor.fit(input, target, options)

    def predict_transform(self, x, mean_flag=True):  # returns the mean
        if x.ndim == 1:
            _trans_x = self.scaler.transform([x])
        else:
            _trans_x = self.scaler.transform(x)

        if mean_flag:
            return self.predict_mean(_trans_x).squeeze()

        else:
            result = self.predict(_trans_x)
            rshape = result.shape

            return result.reshape((rshape[0], rshape[1]))


class XGB_Regressor(Regressor):

    def __init__(self, options):
        self.options = options
        self.regressor = None
        self.feature_importances_ = None
        self.scaler = options['scaler']

    def fit(self, inputs_train, labels_train, fit_options={}):
        xgb_reg = XGBRegressor(random_state=self.options['seed'])

        print('Starting with low learning rate and tuning: \
            max_depth, min_child_weight, n_estimators')

        params = {
            "learning_rate":     [0.1],  # np.arange(0.05,0.45,0.05), #eta
            # np.arange(2,14,2),
            "max_depth":         self.options['max_depth'],
            # np.arange(1,7,6),
            "min_child_weight":  self.options['min_child_weight'],
            # np.arange(10,80,10),
            "n_estimators":      self.options['n_estimators'],

            "colsample_bytree":  [0.8],
            "subsample":         [0.8],
            "gamma":             [0],
        }

        GSCV = GridSearchCV(xgb_reg,  # , #np.arange(0.05,0.45,0.05), #eta),
                            params,
                            cv=self.options['cv'],
                            scoring=self.options['scoring'],
                            n_jobs=self.options['n_jobs'],
                            verbose=self.options['verbose'],  # verbose,
                            return_train_score=True)

        GSCV.fit(inputs_train, labels_train)

        print('best_params_:', GSCV.best_params_)  # ,
        print('best_score_:',  GSCV.best_score_)

        print('Tuning: gamma')
        params = {
            "learning_rate":    [0.1],  # np.arange(0.05,0.45,0.05), #eta
            "max_depth":        [GSCV.best_params_['max_depth']],
            "min_child_weight": [GSCV.best_params_['min_child_weight']],
            "n_estimators":     [GSCV.best_params_['n_estimators']],

            "colsample_bytree": [0.8],
            "subsample":        [0.8],
            # np.arange(0.05,0.45,0.05),
            "gamma":            self.options['gamma'],
        }

        GSCV = GridSearchCV(xgb_reg,  # , #np.arange(0.05,0.45,0.05), #eta),
                            params,
                            cv=self.options['cv'],
                            scoring=self.options['scoring'],
                            n_jobs=self.options['n_jobs'],
                            verbose=self.options['verbose'],  # verbose,
                            return_train_score=True)

        GSCV.fit(inputs_train, labels_train)

        print('best_params_:', GSCV.best_params_)  # ,
        print('best_score_:', GSCV.best_score_)

        print('Tuning: colsample_bytree, subsample')

        params = {
            "learning_rate":    [0.1],  # np.arange(0.05,0.45,0.05), #eta
            "max_depth":        [GSCV.best_params_['max_depth']],
            "min_child_weight": [GSCV.best_params_['min_child_weight']],
            "n_estimators":     [GSCV.best_params_['n_estimators']],
            "gamma":            [GSCV.best_params_['gamma']],

            # np.arange(0.60, 0.95, 0.05),
            "colsample_bytree": self.options['colsample_bytree'],
            # np.arange(0.60, 0.95, 0.05),
            "subsample":        self.options['subsample'],
        }

        GSCV = GridSearchCV(xgb_reg,  # , #np.arange(0.05,0.45,0.05), #eta),
                            params,
                            cv=self.options['cv'],
                            scoring=self.options['scoring'],
                            n_jobs=self.options['n_jobs'],
                            verbose=self.options['verbose'],  # verbose,
                            return_train_score=True)

        GSCV.fit(inputs_train, labels_train)

        print('best_params_:', GSCV.best_params_)  # ,
        print('best_score_:', GSCV.best_score_)

        print('Tuning: reg_alpha, reg_lambda')

        params = {
            "learning_rate":    [0.1],  # np.arange(0.05,0.45,0.05), #eta
            "max_depth":        [GSCV.best_params_['max_depth']],
            "min_child_weight": [GSCV.best_params_['min_child_weight']],
            "n_estimators":     [GSCV.best_params_['n_estimators']],
            "gamma":            [GSCV.best_params_['gamma']],

            "colsample_bytree": [GSCV.best_params_['colsample_bytree']],
            "subsample":        [GSCV.best_params_['subsample']],


            # ,[1e-5, 1e-2, 0.1, 1, 10], #alpha
            "reg_alpha":        self.options['reg_alpha'],
            # [1e-5, 1e-2, 0.1, 1, 10],#lambda
            "reg_lambda":       self.options['reg_lambda'],
        }

        GSCV = GridSearchCV(xgb_reg,  # , #np.arange(0.05,0.45,0.05), #eta),
                            params,
                            cv=self.options['cv'],
                            scoring=self.options['scoring'],
                            n_jobs=self.options['n_jobs'],
                            verbose=self.options['verbose'],  # verbose,
                            return_train_score=True)

        GSCV.fit(inputs_train, labels_train)

        print('best_params_:', GSCV.best_params_)  # ,
        print('best_score_:', GSCV.best_score_)

        print('Tuning: learning_rate')

        params = {
            # np.arange(0.025,0.150,0.025), #np.arange(0.05,0.45,0.05), #eta
            "learning_rate":    self.options['learning_rate'],
            "max_depth":        [GSCV.best_params_['max_depth']],
            "min_child_weight": [GSCV.best_params_['min_child_weight']],
            "n_estimators":     [GSCV.best_params_['n_estimators']],
            "gamma":            [GSCV.best_params_['gamma']],

            "colsample_bytree": [GSCV.best_params_['colsample_bytree']],
            "subsample":        [GSCV.best_params_['subsample']],


            "reg_alpha":        [GSCV.best_params_['reg_alpha']],  # alpha
            "reg_lambda":       [GSCV.best_params_['reg_lambda']]  # lambda
        }

        GSCV = GridSearchCV(xgb_reg,  # , #np.arange(0.05,0.45,0.05), #eta),
                            params,
                            cv=self.options['cv'],
                            scoring=self.options['scoring'],
                            n_jobs=self.options['n_jobs'],
                            verbose=self.options['verbose'],  # verbose,
                            return_train_score=True)

        GSCV.fit(inputs_train, labels_train)

        print('best_params_:', GSCV.best_params_)  # ,
        print('best_score_:', GSCV.best_score_)

        print('Final model')

        # Regression
        regressor = XGBRegressor(random_state=self.options['seed'])  # seed)
        regressor.set_params(**GSCV.best_params_)
        trained_regressor = regressor.fit(inputs_train, labels_train)
        self.regressor = trained_regressor
        self.feature_importances_ = self.regressor.feature_importances_

    def predict(self, inputs):
        return self.regressor.predict(inputs)

    def predict_transform(self, x):
        ''' predictor function (trained regressor) '''

        x = np.asarray(x)
        ndim = x.ndim

        if ndim == 1:
            x = x.reshape(1, -1)

        _trans_x = self.scaler.transform(x)
        result = self.regressor.predict(_trans_x)

        if ndim == 1:
            result = result[0]
        return result
