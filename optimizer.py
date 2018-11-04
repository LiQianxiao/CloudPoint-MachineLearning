'''
File: optimizer.py
File Created: Sunday, 4th November 2018 7:34:47 pm
Author: Qianxiao Li (liqix@ihpc.a-star.edu.sg)
        Gonzalez Oyarce Anibal Lautaro (anibal-gonza@ihpc.a-star.edu.sg)
-----
License: MIT License
'''

import numpy as np
from pyswarm import pso
import pandas as pd
import time
import multiprocessing as mp


def unwrap_self(arg, **kwarg):
    return PSO_Optimizer.optimisation(*arg, **kwarg)


class PSO_Optimizer(object):

    def __init__(self, regressor, scaler, objective,
                 constraints, selection_criteria):
        self.regressor = regressor
        self.scaler = scaler
        self.objective = lambda x, cp: objective(x, cp, regressor)
        self.constraints = constraints
        self.selection_criteria = selection_criteria

    def optimisation(self, n_run, optimisation_options, manager_list):
        print('Optimization No.:', n_run)
        np.random.seed(n_run)

        xopt, fopt = pso(
            self.objective,
            optimisation_options['lb'],
            optimisation_options['ub'],
            f_ieqcons=self.constraints,
            args=[optimisation_options['target_CP']],
            swarmsize=optimisation_options['swarmsize'],
            omega=optimisation_options['omega'],
            phip=optimisation_options['phip'],
            phig=optimisation_options['phig'],
            maxiter=optimisation_options['maxiter'],
            debug=optimisation_options['debug_flag'])
        manager_list.append([xopt, fopt])
        return (xopt, fopt)

    def optimisation_parallel(self, optimisation_options):
        n_solutions = optimisation_options['n_solutions']
        nprocessors = optimisation_options['nprocessors']
        start = time.time()

        print('Optimisation sequence started at time:')
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(start)))

        run_flag = True
        round_count = 0

        sols_per = []
        sols_xopts = []
        sols_pred = []

        while run_flag:
            seeds_list = np.arange(1, nprocessors+1) + nprocessors*round_count

            round_count += 1
            print('Working on round:', round_count)

            manager = mp.Manager()
            return_list = manager.list()

            processes = []
            for i in seeds_list:
                p = mp.Process(
                    target=unwrap_self,
                    args=([self, i, optimisation_options, return_list], ))
                processes.append(p)

            [x.start() for x in processes]
            [x.join() for x in processes]

            results_optimisation = np.array(return_list)

            xopts = results_optimisation[:, 0]
            fopts = results_optimisation[:, 1]

            preds_xopts = np.array(
                [self.regressor.predict_transform(item) for item in xopts])

            per_sol_preds = abs(
                preds_xopts - optimisation_options['target_CP']) \
                / optimisation_options['target_CP']

            for _xopt, _pred_xopt, _per in zip(
                    xopts, preds_xopts, per_sol_preds):

                if _per <= optimisation_options['criteria']:
                    if self.selection_criteria:
                        if self.selection_criteria(_xopt):
                            sols_per.append(_per)
                            sols_xopts.append(_xopt)
                            sols_pred.append(_pred_xopt)

                    else:
                        sols_per.append(_per)
                        sols_xopts.append(_xopt)
                        sols_pred.append(_pred_xopt)

            print('So far', len(sols_per), 'solutions have been found')
            if len(sols_per) >= n_solutions:
                run_flag = False

                sols_per = np.array(sols_per)
                sols_xopts = np.array(sols_xopts)
                sols_pred = np.array(sols_pred)

                sols_per_argsort = np.array(sols_per).argsort()

                sols_per = sols_per[sols_per_argsort]
                sols_xopts = sols_xopts[sols_per_argsort]
                sols_pred = sols_pred[sols_per_argsort]

                sols_per = sols_per[:n_solutions]
                sols_xopts = sols_xopts[:n_solutions]
                sols_pred = sols_pred[:n_solutions]

            if round_count > optimisation_options['max_rounds']:
                run_flag = False

        sols_df = pd.DataFrame()

        for _item in enumerate(optimisation_options['opt_vars']):
            sols_df[optimisation_options['opt_vars'][_item[0]]] = \
                sols_xopts[:, _item[0]]

        sols_df['Cloud Point'] = sols_pred
        sols_df['% Error'] = sols_per

        end = time.time()
        runtime = end - start

        print('Optimisation finished in %.3f' % (runtime), '[s]')

        sols_df.columns = ['Pred_'+item for item in sols_df.columns]

        sols_df.loc[:, 'target_CP'] = optimisation_options['target_CP']

        return sols_df, runtime
