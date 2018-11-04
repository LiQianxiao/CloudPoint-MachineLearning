'''
File: utils_plot.py
File Created: Sunday, 4th November 2018 7:34:47 pm
Author: Qianxiao Li (liqix@ihpc.a-star.edu.sg)
        Gonzalez Oyarce Anibal Lautaro (anibal-gonza@ihpc.a-star.edu.sg)
-----
License: MIT License
'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred):
    """Returns the root-mean-squared-error

    Arguments:
        y_true {array-like} -- true target values
        y_pred {array-like} -- predicted values

    Returns:
        float -- rmse
    """
    return sqrt(mean_squared_error(y_true, y_pred))


def plot_comparisons(targets, reg_preds, size=(5, 5), save_path=None):
    """
    Plot comparison
    """
    matplotlib.rcParams.update({'font.size': 16})
    alpha = 1.0
    y_min = 0
    y_max = 100
    lims = (y_min, y_max)
    ticks = np.linspace(y_min, y_max, 5)

    n_plots = len(reg_preds)
    n_cols = n_plots

    fig, axes = plt.subplots(
        1, n_plots, figsize=(size[0]*n_plots, size[1]),
        squeeze=False, sharex=True, sharey=False)

    if not isinstance(targets, list):
        targets = [targets, ]*n_plots

    for ax, pred, target in zip(axes[0, :], reg_preds, targets):
        # Scatter plot
        if 'train' in pred:
            train_label = r'Train (RMSE=%.1f$^\circ$C)' % rmse(
                target['train'], pred['train'])
            ax.scatter(
                target['train'], pred['train'],
                c='k', marker='o', alpha=alpha, label=train_label)

        if 'valid' in pred:
            valid_label = r'Validation (RMSE=%.1f$^\circ$C)' % rmse(
                target['valid'], pred['valid'])
            ax.scatter(
                target['valid'], pred['valid'],
                c='b', marker='v', alpha=alpha, label=valid_label)

        if 'test' in pred:
            test_label = r'Test (RMSE=%.1f$^\circ$C)' % rmse(
                target['test'], pred['test'])
            ax.scatter(
                target['test'], pred['test'],
                c='r', marker='v', alpha=alpha, label=test_label)

        # Line
        ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)

        # Formatting
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title(pred['name'])
        ax.set_xlabel(r'Target Cloud Point ($^\circ$C)')
        ax.set_ylabel(r'Predicted Cloud Point ($^\circ$C)')
        ax.legend(loc=4, prop={'size': 11})

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print('Figure saved to: %s' % save_path)


def plot_predictions(y, y_hat, labels, save_path=None):
    """Scatter plots for prediction vs true values

    Arguments:
        y {list of array-like} -- list of targets
        y_hat {list of array-like} -- list of predictions
        labels {list of str} -- list of labels

    Keyword Arguments:
        save_path {str} -- path to save figure to (default: {None})
    """
    matplotlib.rcParams.update({'font.size': 22})
    y, y_hat, labels = list(
        map(
            lambda l: l if isinstance(l, list) else [l],
            [y, y_hat, labels]))
    n_plots = len(y)
    y_min = min([min(z) for z in y])
    y_max = max([max(z) for z in y])
    lims = (y_min, y_max)
    fig, ax = plt.subplots(
        1, n_plots, figsize=(10*n_plots, 8),
        squeeze=False, sharex=True, sharey=True)
    for axis, target, prediction, label in zip(ax[0, :], y, y_hat, labels):
        # Scatter plot
        axis.scatter(target, prediction, alpha=0.3)

        # Title and labels
        rmse_value = rmse(target, prediction)
        title = label + r" (RMSE=%.1f$^\circ$C)" % rmse_value
        axis.set_title(title)
        axis.set_xlabel(r'Target Cloud Point ($^\circ$C)')
        axis.set_ylabel(r'Predicted Cloud Point ($^\circ$C)')
        axis.plot(lims, lims, 'k--', alpha=0.75, zorder=0)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print('Figure saved to: %s' % save_path)


def plot_feature_importances(importances, columns, xerrs=None, save_path=None):
    """plot feature importances

    Arguments:
        importances {list of float} -- feature importances
        columns {list of str} -- labels

    Keyword Arguments:
        save_path {str} -- path to save figure to (default: {None})
    """

    matplotlib.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax = plt.axes()
    coords = np.arange(len(importances))
    ax.set_yticks(coords)
    ax.set_yticklabels(columns)
    ax.set_xlabel("Feature Importance (bigger = more important)")
    # ax.set_title('Feature Importances')
    ax.barh(
        coords, importances,
        xerr=xerrs, capsize=10.0, ecolor='k',
        edgecolor='k', height=0.8, color='skyblue', alpha=1.0)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print('Figure saved to: %s' % save_path)
