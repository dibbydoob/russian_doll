import pytz
import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from quant_stats import stepdown_algorithm
from copy import deepcopy
from datetime import datetime
from alpha import Alpha
import quant_stats
# from quantyhlib.general_utils import save_file, load_file
import random
from quant_stats import marginal_family_test

from data_master import DataMaster

class EQMOM(Alpha):

    def __init__(
        self, 
        trade_range=None,
        instruments=[], 
        execrates=None,
        commrates=None,
        longswps=None,
        shortswps=None, 
        dfs={},  
        positional_inertia=0
    ):
        super().__init__(
            trade_range=trade_range, 
            instruments=instruments, 
            execrates=execrates,
            commrates=commrates,
            longswps=longswps,
            shortswps=shortswps,
            dfs=dfs,
            positional_inertia=positional_inertia 
        )

    def param_generator(self, shattered):
        return super().param_generator(shattered=shattered)

    async def compute_signals_unaligned(self, shattered=True, param_idx=0, index=None):
        alphas = []
        for inst in self.instruments:
            print(inst)
            self.dfs[inst]["smaf"] = self.dfs[inst]["close"].rolling(10).mean()
            self.dfs[inst]["smam"] = self.dfs[inst]["close"].rolling(30).mean()
            self.dfs[inst]["smas"] = self.dfs[inst]["close"].rolling(100).mean()
            self.dfs[inst]["smass"] = self.dfs[inst]["close"].rolling(300).mean()
            inst_alpha = 0.0 + \
                (self.dfs[inst]["smaf"] > self.dfs[inst]["smam"]) + \
                (self.dfs[inst]["smaf"] > self.dfs[inst]["smas"]) + \
                (self.dfs[inst]["smaf"] > self.dfs[inst]["smass"])+ \
                (self.dfs[inst]["smam"] > self.dfs[inst]["smas"]) + \
                (self.dfs[inst]["smam"] > self.dfs[inst]["smass"])
            alphas.append(inst_alpha)
        
        alphadf = pd.concat(alphas, axis=1)
        alphadf.columns = self.instruments
        self.pad_ffill_dfs["alphadf"] = alphadf
        return

    async def compute_signals_aligned(self, shattered=True, param_idx=0, index=None):
        return 
    
    '''Expose strategy custom signal dfs, invriskdf, eligiblesdf'''
    def instantiate_eligibilities_and_strat_variables(self, delta_lag=0):
        eligibles = []
        for inst in self.instruments:
            inst_eligible = (~np.isnan(self.dfs[inst]["smass"])) \
                & self.activedf[inst] \
                & (self.voldf[inst] > 0).astype("bool") \
                & self.baseclosedf[inst] > 0
            eligibles.append(inst_eligible)
        self.alphadf = self.pad_ffill_dfs["alphadf"]
        self.invriskdf = np.log(1 / self.voldf) / np.log(1.3)
        self.eligiblesdf = pd.concat(eligibles, axis=1)
        self.eligiblesdf.columns = self.instruments
        self.eligiblesdf.astype("int8")
        self.alphadf = self.alphadf.shift(delta_lag).fillna(0)
        self.eligiblesdf *= self.eligiblesdf.shift(delta_lag).fillna(0)
        return

    def compute_forecasts(self, portfolio_i, date, eligibles_row):
        return self.alphadf.loc[date], np.sum(eligibles_row)

    def post_risk_management(self, nominal_tot, positions, weights, eligibles_i=None, eligibles_row=None, *args, **kwargs):
        return nominal_tot, positions, weights

class PairMom(Alpha):

    def __init__(
        self, 
        slow, fast,
        trade_range=None,
        instruments=[], 
        execrates=None,
        commrates=None,
        longswps=None,
        shortswps=None, 
        dfs={},  
        positional_inertia=0
    ):  
        self.fast = fast
        self.slow = slow
        super().__init__(
            trade_range=trade_range, 
            instruments=instruments, 
            execrates=execrates,
            commrates=commrates,
            longswps=longswps,
            shortswps=shortswps,
            dfs=dfs,
            positional_inertia=positional_inertia 
        )

    def get_fast_slow_pair(self):
        return self.fast, self.slow

    def param_generator(self, shattered):
        return super().param_generator(shattered=shattered)

    async def compute_signals_unaligned(self, shattered=True, param_idx=0, index=None):
        alphas = []
        for inst in self.instruments:
            print(inst)
            inst_alpha = 0.0 + \
                self.dfs[inst]["close"].rolling(self.get_fast_slow_pair()[0]).mean() \
                > self.dfs[inst]["close"].rolling(self.get_fast_slow_pair()[1]).mean()
            alphas.append(inst_alpha)
        
        alphadf = pd.concat(alphas, axis=1)
        alphadf.columns = self.instruments
        self.pad_ffill_dfs["alphadf"] = alphadf
        return

    async def compute_signals_aligned(self, shattered=True, param_idx=0, index=None):
        return 
    
    '''Expose strategy custom signal dfs, invriskdf, eligiblesdf'''
    def instantiate_eligibilities_and_strat_variables(self, delta_lag=0):
        eligibles = []
        for inst in self.instruments:
            slow_rolling = self.dfs[inst]["close"].rolling(self.get_fast_slow_pair()[1]).mean()
            inst_eligible = (~np.isnan(slow_rolling)) \
                & self.activedf[inst] \
                & (self.voldf[inst] > 0).astype("bool") \
                & self.baseclosedf[inst] > 0
            eligibles.append(inst_eligible)
        self.alphadf = self.pad_ffill_dfs["alphadf"]
        self.invriskdf = np.log(1 / self.voldf) / np.log(1.3)
        self.eligiblesdf = pd.concat(eligibles, axis=1)
        self.eligiblesdf.columns = self.instruments
        self.alphadf = self.alphadf.shift(delta_lag).fillna(0)
        self.eligiblesdf *= self.eligiblesdf.shift(delta_lag).fillna(0)
        return

    def compute_forecasts(self, portfolio_i, date, eligibles_row):
        return self.alphadf.loc[date], np.sum(eligibles_row)

    def post_risk_management(self, nominal_tot, positions, weights, eligibles_i=None, eligibles_row=None, *args, **kwargs):
        return nominal_tot, positions, weights

async def main():
    load_dump = True
    if not load_dump:
        data_master = DataMaster()
        misc_service = data_master.get_misc_service()
        index_service = data_master.get_indices_service()
        equity_service = data_master.get_equity_service()

        comps = index_service.get_index_components("GSPC")
        index_components = list(comps["Code"])
        period_start = datetime(1990, 1, 1, tzinfo=pytz.utc)
        period_end = datetime.now(pytz.utc)
        
        component_data = await equity_service.asyn_batch_get_ohlcv(
            tickers=index_components, 
            read_db=False, 
            insert_db=False, 
            granularity="d", 
            engine="eodhistoricaldata",
            period_start=period_start,
            period_end=period_end
        )
        print(component_data)
        save_file("temp.dump", (index_components, component_data, period_start, period_end))
    else:
        (index_components, component_data, period_start, period_end) = load_file("temp.dump")

    universe_size = 10
    index_components = index_components[:universe_size]
    component_data = component_data[:universe_size]



    '''Run Solo Hypothesis Tests''' 
    dfs = {
        comp + "%USD" : 
        data.reset_index(drop=True).set_index("datetime") 
        for comp, data in zip(index_components, component_data) 
    }

    trade_insts = [k + "%USD" for k in index_components]
    eqmom = EQMOM(
        trade_range=(period_start, period_end),
        instruments=trade_insts,  
        dfs=dfs,  
        positional_inertia=0
    )
    await eqmom.run_simulation(verbose=True)
    save_file("eqmom.dump", eqmom)
    await eqmom.write_stats(children=False, dat_shuffles=20)
 
    '''Run Family Tests'''
    def member_generator(seed, datacopy, slow, fast):
        alpha_params = {
            "slow": slow, "fast": fast,
            "trade_range": (period_start, period_end), 
            "instruments":trade_insts, 
            "positional_inertia":0
        }
        alpha_params.update({"dfs" : datacopy})
        return seed(**alpha_params)

    def performance_criterion(rets, leverages, weights):
        capital_ret = [
            lev_scalar * np.dot(weight, ret) 
            for lev_scalar, weight, ret in zip(leverages.values, rets.values, weights.values)
        ]
        sharpe = np.mean(capital_ret) / np.std(capital_ret) * np.sqrt(253)
        return sharpe

    n = 10
    family_seed = [PairMom] * n

    pairs = [random.choices(list(range(5, 200)), k=2) for _ in range(n)]
    fast_slows = [(min(pair), max(pair)) for pair in pairs]
    alpha_family = [
        member_generator(seed, deepcopy(dfs), fast=pair[0], slow=pair[1]) 
        for seed, pair in zip(family_seed, fast_slows)
    ]
    family_ps, exactness = await marginal_family_test(
        criterion_function=performance_criterion, 
        alpha_family=alpha_family,
        m=100
    )
    print(family_ps)
    print(exactness)

async def hyothesis_tests_tester():

    
    criterion_function = lambda sample: np.mean(sample) #identity transform
    norm_ret_gen = lambda: [np.random.normal(0, 16) for _ in range(5000)] #~20 years of trading days
    perf_generator = lambda: [np.random.normal(1, 16) for _ in range(5000)] #true Sharpe 1 trading strategy
    famsize = 10 #family size
    rounds = 500
    true_sig = not_sig = int(famsize / 2)
    
    alpha = 0.05
    def family_test():
        unperm_criterions = []

        for member in range(true_sig):  
            unperm_criterions.append(criterion_function(perf_generator()))
        for member in range(not_sig):
            unperm_criterions.append(criterion_function(norm_ret_gen()))

        def round_criterions():
            round = []
            for member in range(famsize):
                round.append(criterion_function(norm_ret_gen()))
            return round

        round_stats = []
        for round in range(rounds):
            round_stats.append(round_criterions())

        p, e = stepdown_algorithm(
            unpermuted_criterions=unperm_criterions, 
            round_criterions=round_stats, 
            alpha=alpha
        )
        print(p, e)
        return p

    fer_errors = []
    for _ in range(100):
        p = family_test()
        significant = p < alpha
        fer_err = np.sum(significant[true_sig:].astype('int32')) > 0
        fer_errors.append(fer_err)
        print(fer_err)
    
    print(np.sum(np.array(fer_errors).astype("int32"))) #obtained 3
    save_file("fer.data", fer_errors)
    return

if __name__ == "__main__":
    strat_demo = False
    if strat_demo: asyncio.run(main())
    else: asyncio.run(hyothesis_tests_tester())