import pytz
import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from alpha import Alpha
from alpha import Amalgapha
# from general_utils import save_file, load_file
from data_master import DataMaster


class MNREV(Alpha):

    def __init__(
        self, 
        trade_range=None,
        instruments=[], 
        execrates=None,
        commrates=None,
        longswps=None,
        shortswps=None, 
        dfs={},  
        positional_inertia=0,
        strat_configs= {
            "shortrate": 0.10,
            "longrate": 0.10
        }
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
        self.strat_configs = strat_configs

    def set_strat_configs(self, strat_configs):
        self.strat_configs = strat_configs

    def _check_configs(self, strat_configs):
        assert "shortrate" in strat_configs and "longrate" in strat_configs
 
    def param_generator(self, shattered, param_idx=0):
        axials = []
        for dd_lookback in range(30, 260, 10):
            axials.append((dd_lookback,))
        if shattered:
            def yield_params():
                for param in axials:
                    yield param
            return yield_params()
        else:
            return [axials[param_idx]]

    async def compute_metas(self, index, delta_lag, shattered=True, param_idx=0):
        await super().compute_metas(index)
        self._check_configs(self.strat_configs)

        dds, eligibles = [], []
        for inst in self.instruments:
            print(inst)
            
            dd_lookbacks = []
            for params in self.param_generator(shattered=shattered, param_idx=param_idx):
                ddi = \
                    (1 + self.dfs[inst]["ret"]).rolling(window=params[0]).apply(np.prod, raw=True, engine="numba") \
                    / (1 + self.dfs[inst]["ret"]).rolling(window=params[0]).apply(np.prod, raw=True, engine="numba").cummax() \
                    - 1
                dd_lookbacks.append(ddi)
            self.dfs[inst]["dd"] = pd.concat(dd_lookbacks, axis=1).mean(axis=1)
            self.dfs[inst]["eligible"] = \
                (~np.isnan(self.dfs[inst]["dd"])) \
                & self.dfs[inst]["active"] \
                & (self.dfs[inst]["vol"] > 0) \
                & (self.dfs[inst]["adj_close"] > 0) \
            
            eligibles.append(self.dfs[inst]["eligible"])
            dds.append(self.dfs[inst]["dd"])

        self.invriskdf = np.log(1 / self.voldf) / np.log(1.3)
        self.eligiblesdf = pd.concat(eligibles, axis=1)
        self.eligiblesdf.columns = self.instruments
        self.dddf = pd.concat(dds, axis=1)
        self.dddf.columns = self.instruments
        
        self.eligiblesdf = self.eligiblesdf.shift(delta_lag).fillna(0)
        self.dddf = self.dddf.shift(delta_lag).fillna(0)

    def compute_forecasts(self, portfolio_i, date, eligibles_row):
        drawdowns = self.dddf.loc[date].values
        eligible_args = np.where(eligibles_row == 1)[0]
        eligible_alphas = np.take(-1 * drawdowns, eligible_args)
        argsort_alphas = np.argsort(eligible_alphas)
        eligibles_size = np.sum(eligibles_row)
        shortsize = int(eligibles_size * self.strat_configs["shortrate"])
        longsize = int(eligibles_size * self.strat_configs["longrate"])
        shorts = np.take(eligible_args, argsort_alphas[:shortsize]).astype("int32")
        longs = np.take(eligible_args, argsort_alphas[-longsize:]).astype("int32")
        forecasts = np.zeros(len(eligibles_row))
        for i in shorts:
            forecasts[i] = -1
        for i in longs:
            forecasts[i] = 1
        return forecasts, shortsize + longsize

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

    async def compute_metas(self, index, delta_lag, shattered=True, param_idx=0):
        await super().compute_metas(index)

        alphas, eligibles = [], []
        for inst in self.instruments:
            print(inst)
            self.dfs[inst]["smaf"] = self.dfs[inst]["adj_close"].rolling(10).mean()
            self.dfs[inst]["smam"] = self.dfs[inst]["adj_close"].rolling(30).mean()
            self.dfs[inst]["smas"] = self.dfs[inst]["adj_close"].rolling(100).mean()
            self.dfs[inst]["smass"] = self.dfs[inst]["adj_close"].rolling(300).mean()
            self.dfs[inst]["alphas"] = 0.0 + \
                (self.dfs[inst]["smaf"] > self.dfs[inst]["smam"]) + \
                (self.dfs[inst]["smaf"] > self.dfs[inst]["smas"]) + \
                (self.dfs[inst]["smaf"] > self.dfs[inst]["smass"])+ \
                (self.dfs[inst]["smam"] > self.dfs[inst]["smas"]) + \
                (self.dfs[inst]["smam"] > self.dfs[inst]["smass"])

            self.dfs[inst]["eligible"] = \
                (~np.isnan(self.dfs[inst]["smass"])) \
                & self.dfs[inst]["active"] \
                & (self.dfs[inst]["vol"] > 0) \
                & (self.dfs[inst]["adj_close"] > 0) \
            
            alphas.append(self.dfs[inst]["alphas"])
            eligibles.append(self.dfs[inst]["eligible"])

        self.invriskdf = np.log(1 / self.voldf) / np.log(1.3)
        self.alphadf = pd.concat(alphas, axis=1)
        self.alphadf.columns = self.instruments
        self.eligiblesdf = pd.concat(eligibles, axis=1)
        self.eligiblesdf.columns = self.instruments
        self.eligiblesdf.astype('int8')

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
            read_db=True, 
            insert_db=True, 
            granularity="d", 
            engine="eodhistoricaldata",
            period_start=period_start,
            period_end=period_end
        )
        print(component_data)
        save_file("temp.dump", (index_components, component_data, period_start, period_end))
    else:
        (index_components, component_data, period_start, period_end) = load_file("temp.dump")

    subset = 100
    index_components = index_components[:subset]
    component_data = component_data[:subset]    
    dfs = {
        comp + "%USD" : data.reset_index(drop=True).set_index("datetime") for comp, data in zip(index_components, component_data) 
    }

    trade_insts = [k + "%USD" for k in index_components]
    eqmom = EQMOM(
        trade_range=(period_start, period_end),
        instruments=trade_insts,  
        dfs=dfs,  
        positional_inertia=0
    )
    eqrev = MNREV(
        trade_range=(period_start, period_end),
        instruments=trade_insts,  
        dfs=dfs,  
        positional_inertia=0
    )
    alphas = [eqmom, eqrev]
    strat_dfs = [await alpha.run_simulation(verbose=True, delta_lag=0) for alpha in alphas]
    stats = [await alpha.write_stats() for alpha in alphas]

    leveragess = [alpha.get_leverages() for alpha in alphas]
    weightss = [alpha.get_weights() for alpha in alphas]
    instrumentss = [alpha.get_instruments() for alpha in alphas]
    
    amalgapha = Amalgapha(
        trade_range=(period_start, period_end),
        instruments=trade_insts,
        dfs=dfs,
        positional_inertia=0.20,
        weightss=weightss,
        leveragess=leveragess,
        execrates=np.array([0.0014753723651389232] * len(trade_insts)),
        commrates=np.zeros(len(trade_insts)),
        longswps=np.zeros(len(trade_insts)),
        shortswps=np.zeros(len(trade_insts)),
    )

    combined_strat_df = await amalgapha.run_simulation(verbose=True)
    await amalgapha.write_stats()

    log_rets = lambda daily_ret_ser: np.log((1 + daily_ret_ser).cumprod())
    plt.plot(log_rets(strat_dfs[0]["capital_ret"]), label="momentum")
    plt.plot(log_rets(strat_dfs[1]["capital_ret"]), label="mean_rev")
    plt.plot(log_rets(combined_strat_df["capital_ret"]), label="combination")
    plt.legend()
    plt.savefig("temp.png")
    plt.close()

if __name__ == "__main__":
    asyncio.run(main())