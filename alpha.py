import pytz
import numpy as np 
import pandas as pd
from numba import jit
from pprint import pprint
from copy import deepcopy
from datetime import datetime

from abc import ABC
from abc import abstractmethod

import db_logs as db_logs

def get_pnl_stats(last_weights, last_units, baseclose_prev, ret_row, leverages):
    ret_row = np.nan_to_num(ret_row, nan=0, posinf=0, neginf=0)
    day_pnl = np.sum(last_units * baseclose_prev * ret_row)
    nominal_ret = np.dot(last_weights, ret_row)
    capital_ret = nominal_ret * leverages[-1]
    return day_pnl, nominal_ret, capital_ret

class BaseAlpha(ABC):
    '''Accepts instruments with fx code %attached'''
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
        self.trade_range = trade_range
        self.instruments = instruments
        self.execrates = execrates if execrates is not None else np.zeros(len(instruments))
        self.commrates = commrates if commrates is not None else np.zeros(len(instruments))
        self.longswps = longswps if longswps is not None else np.zeros(len(instruments))
        self.shortswps = shortswps if shortswps is not None else np.zeros(len(instruments))
        self.dfs = dfs
        self.positional_inertia = positional_inertia
    
    def set_instruments_settings(self, instruments, execrates=None, commrates=None, longswps=None, shortswps=None):
        self.instruments = instruments
        self.execrates = execrates if execrates is not None else np.zeros(len(instruments))
        self.commrates = commrates if commrates is not None else np.zeros(len(instruments))
        self.longswps = longswps if longswps is not None else np.zeros(len(instruments))
        self.shortswps = shortswps if shortswps is not None else np.zeros(len(instruments))
        
    def set_dfs(self, dfs):
        self.dfs = dfs
    
    def get_instruments(self):
        return self.instruments

    def get_weights(self):
        return self.weights_df
    
    def get_positions(self):
        return self.positions_df

    def _get_targets(self):
        return self.targets_df

    def get_inst_last_targets(self, adjust_to_capital=None):
        scalar = 1
        if adjust_to_capital:
            scalar = adjust_to_capital / self.get_capitals().values[-1]
        last_targets = self._get_targets().values[-1]
        result = {}
        for i in range(len(self.instruments)):
            result[self.instruments[i]] = scalar * last_targets[i] 
        return result
    
    def get_capitals(self):
        return self.capital_ser
    
    def get_leverages(self):
        return self.leverages_ser
    
    def get_last_closes(self, fx_adjusted=False):
        if fx_adjusted:
            adj_closes = self.baseclosedf.values[-1]
            return {self.instruments[i]: adj_closes[i] for i in range(len(self.instruments))} 
        closes = self.closedf.values[-1]
        return {self.instruments[i] : closes[i] for i in range(len(self.instruments))}

    async def write_stats(self, children=False):
        assert self.portfolio_df is not None
        '''do statistical analysis, bootstrapping tests, shattering analysis etc'''
        def stats_on_df(portfolio_df):
            tradeful_df = portfolio_df.loc[portfolio_df["capital_ret"] != 0]
            costless_rets = tradeful_df["capital_ret"]
            swapful_rets = costless_rets - tradeful_df["swap_penalty"]
            execful_rets = costless_rets - tradeful_df["exec_penalty"]
            commful_rets = costless_rets - tradeful_df["comm_penalty"]
            costful_rets = costless_rets - tradeful_df["comm_penalty"] - tradeful_df["exec_penalty"] - tradeful_df["swap_penalty"] 
            costless_sharpe = np.mean(costless_rets.values) / np.std(costless_rets.values) * np.sqrt(253)
            swapful_sharpe = np.mean(swapful_rets.values) / np.std(swapful_rets.values) * np.sqrt(253)
            execful_sharpe = np.mean(execful_rets.values) / np.std(execful_rets.values) * np.sqrt(253)
            commful_sharpe = np.mean(commful_rets.values) / np.std(commful_rets.values) * np.sqrt(253)
            costful_sharpe = np.mean(costful_rets.values) / np.std(costful_rets.values) * np.sqrt(253)
            costdrag_pa = (costless_sharpe - costful_sharpe) * np.std(costless_rets.values) * np.sqrt(253)
        
            return {
                "costdrag_pa": costdrag_pa,
                "sharpe_costful": costful_sharpe,
                "sharpe_costless": costless_sharpe,
                "sharpe_swapful": swapful_sharpe,
                "sharpe_execful": execful_sharpe,
                "sharpe_commful": commful_sharpe,
            }
        stats_dict = {}
        stats = stats_on_df(self.portfolio_df)
        pprint(stats)
        stats_dict["core"] = {
            "df": self.portfolio_df,
            "stats": stats
        }
        if children:
            shattered_params = list(deepcopy(self).param_generator(shattered=True))
            for i in range(len(shattered_params)):
                component_df = await self.run_simulation(shattered=False, param_idx=i)
                stats = stats_on_df(component_df)
                pprint(stats)
                stats_dict[str(shattered_params[i])] = {
                    "df": self.portfolio_df,
                    "stats": stats
                }
        return stats_dict
        
    '''Exposes voldf, retdf, activedf, baseclosedf > child needs to exposre invriskdf, eligiblesdf'''
    async def compute_metas(self, index):
        @jit(nopython=True)
        def numba_any(x):
            return int(np.any(x))

        vols, rets, actives, closes, fxconvs = [], [], [], [], []

        aligner = pd.DataFrame(index=index) 
        for inst in self.instruments: 
            self.dfs[inst]["vol"] = (-1 + self.dfs[inst]["adj_close"] / self.dfs[inst].shift(1)["adj_close"]).rolling(30).std()
            self.dfs[inst] = aligner.join(self.dfs[inst])
            self.dfs[inst] = self.dfs[inst].fillna(method="ffill").fillna(method="bfill")
            self.dfs[inst]["ret"] = -1 + self.dfs[inst]["adj_close"] / self.dfs[inst].shift(1)["adj_close"]
            self.dfs[inst]["sampled"] = self.dfs[inst]["adj_close"] != self.dfs[inst].shift(1)["adj_close"]
            self.dfs[inst]["active"] = self.dfs[inst]["sampled"].rolling(5).apply(numba_any, engine="numba", raw=True).fillna(0)
            vols.append(self.dfs[inst]["vol"])
            rets.append(self.dfs[inst]["ret"])
            actives.append(self.dfs[inst]["active"])
            closes.append(self.dfs[inst]["adj_close"])

        for inst in self.instruments:
            if inst[-3:] == "USD":
                fxconvs.append(pd.Series(index=index, data=np.ones(len(index))))
            elif inst[-3:] + "USD%USD" in self.dfs:
                fxconvs.append(self.dfs[inst[-3:] + "USD%USD"]["adj_close"])
            elif "USD" + inst[-3:] + "%" + inst[-3:] in self.dfs:
                fxconvs.append(1 / self.dfs["USD" + inst[-3:] + "%" + inst[-3:]]["adj_close"])
            else:
                print("no resolution", inst)
                exit()
        
        self.voldf = pd.concat(vols, axis=1)
        self.voldf.columns = self.instruments
        self.retdf = pd.concat(rets, axis=1)
        self.retdf.columns = self.instruments
        self.activedf = pd.concat(actives, axis=1)
        self.activedf.columns = self.instruments
        closedf = pd.concat(closes, axis=1)
        closedf.columns = self.instruments
        fxconvsdf = pd.concat(fxconvs, axis=1)
        fxconvsdf.columns = self.instruments
        self.closedf = closedf
        self.baseclosedf = fxconvsdf * closedf
        pass

    def init_portfolio_settings(self, trade_range):
        self.portfolio_df = pd.DataFrame(index=trade_range) \
            .reset_index() \
            .rename(columns={"index" : "datetime"})            
        return 10000, 0.001, 1, self.portfolio_df
        
    def get_strat_scaler(self, portfolio_vol, ewmas, ewstrats):
        ann_realized_vol = np.sqrt(ewmas[-1] * 252)
        return portfolio_vol / ann_realized_vol * ewstrats[-1]

    @abstractmethod
    def param_generator(self, shattered, param_idx=0):
        return []

    def get_shattered_axis_cardinality(self):
        return len(list(deepcopy(self).param_generator(shattered=True)))

    @abstractmethod
    def set_positions(self, capital, strat_scalar, portfolio_vol, prev_positions, *args, **kwargs):
        pass 

    @abstractmethod
    def post_risk_management(self, nominal_tot, positions, weights, *args, **kwargs):
        return nominal_tot, positions, weights

    @abstractmethod
    def zip_data_generator(self):
        pass

    def get_fees(self, baseclose_row, positions, prev_positions):
        delta_positions = np.abs(positions - prev_positions)
        notional_trade = delta_positions * baseclose_row
        exec_fees = np.linalg.norm(notional_trade * self.execrates, ord=1)
        comm_fees = np.linalg.norm(notional_trade * self.commrates, ord=1)
        
        notional_long_holdings = np.abs(np.where(positions > 0, positions, 0) * baseclose_row)
        notional_short_holdings = np.abs(np.where(positions < 0, positions, 0) * baseclose_row)
        long_swaps = -1 * np.dot(notional_long_holdings, self.longswps) / 365 * 7/5
        short_swaps = -1 * np.dot(notional_short_holdings, self.shortswps) / 365 * 7/5
        swap_fees = long_swaps + short_swaps
        return exec_fees, comm_fees, swap_fees

    def set_weights(self, nominal_tot, positions, baseclose_row):
        nominals = positions * baseclose_row
        weights = np.nan_to_num(nominals / nominal_tot, nan=0, posinf=0, neginf=0)
        return weights

    async def run_simulation(self, verbose=False, delta_lag=0, shattered=True, param_idx=0):
        assert (self.trade_range and self.instruments and self.dfs)
        """
        Settings
        """
        portfolio_vol = 0.20
        trade_start = self.trade_range[0]
        trade_end = self.trade_range[1]
        trade_datetime_range = pd.date_range(
            start=datetime(trade_start.year, trade_start.month, trade_start.day), 
            end=datetime(trade_end.year, trade_end.month, trade_end.day),  
            freq="D",
            tz=pytz.utc
        )

        """
        Compute Metas
        """
        await (self.compute_metas(index=trade_datetime_range, delta_lag=delta_lag, shattered=shattered, param_idx=param_idx))

        """
        Initialisations
        """
        capital, ewma, ewstrat, self.portfolio_df = self.init_portfolio_settings(trade_range=trade_datetime_range)

        baseclose_prev = None
        self.capitals = [capital]
        self.ewmas = [ewma]
        self.ewstrats = [ewstrat] 
        self.capital_rets = [0]
        self.nominal_rets = [0]
        self.nominalss = []
        self.leverages = []
        self.strat_scalars = []
        self.chargeable_feess, self.exec_feess, self.comm_feess, self.swap_feess = [], [], [], [] 
        self.chargeable_penalty, self.exec_penalty, self.comm_penalty, self.swap_penalty = [], [], [], []
        self.units_held, self.weights_held = [], []
        self.targets_log = []
        self.inertia_log = []
        for unzipped in self.zip_data_generator():
            portfolio_i = unzipped["portfolio_i"]
            portfolio_row = unzipped["portfolio_row"]
            ret_i = unzipped["ret_i"]
            ret_row = unzipped["ret_row"]
            baseclose_i = unzipped["baseclose_i"]
            baseclose_row = unzipped["baseclose_row"]
        
            strat_scalar = 2
            
            if portfolio_i != 0:
                strat_scalar = self.get_strat_scaler( 
                    portfolio_vol=portfolio_vol,
                    ewmas=self.ewmas,
                    ewstrats=self.ewstrats
                )

                day_pnl, nominal_ret, capital_ret = get_pnl_stats(
                    last_weights=self.weights_held[-1], 
                    last_units=self.units_held[-1],
                    baseclose_prev=baseclose_prev,
                    ret_row=ret_row,
                    leverages=self.leverages, 
                )

                self.capitals.append(self.capitals[-1] + day_pnl)
                self.ewmas.append(0.06 * (capital_ret**2) + 0.94 * self.ewmas[-1] if nominal_ret != 0 else self.ewmas[-1])
                self.ewstrats.append(0.06 * strat_scalar + 0.94 * self.ewstrats[-1] if nominal_ret != 0 else self.ewstrats[-1])
                self.nominal_rets.append(nominal_ret)
                self.capital_rets.append(capital_ret)
            
            self.strat_scalars.append(strat_scalar)

            positions, targets, nominal_tot, inertias = self.set_positions(
                capital=self.capitals[-1],
                strat_scalar=self.strat_scalars[-1], 
                portfolio_vol=portfolio_vol,
                prev_positions=self.units_held[-1] if portfolio_i != 0 else np.zeros(len(self.instruments)),
                **unzipped
            )
            weights = self.set_weights(nominal_tot, positions, baseclose_row)

            nominal_tot, positions, weights = self.post_risk_management(
                nominal_tot=nominal_tot, 
                positions=positions, 
                weights=weights,
                **unzipped
            )

            exec_fees, comm_fees, swap_fees = self.get_fees(
                baseclose_row=baseclose_row,
                positions=positions,
                prev_positions=self.units_held[-1] if portfolio_i != 0 else np.zeros(len(self.instruments))
            ) if self.capital_rets[-1] != 0 else (0, 0, 0)
            
            self.exec_feess.append(exec_fees)
            self.exec_penalty.append(exec_fees / self.capitals[-1])
            self.comm_feess.append(comm_fees)
            self.comm_penalty.append(comm_fees / self.capitals[-1])
            self.swap_feess.append(swap_fees)
            self.swap_penalty.append(swap_fees / self.capitals[-1])

            chargeable_fees = exec_fees + comm_fees + swap_fees
            self.chargeable_feess.append(chargeable_fees)
            self.chargeable_penalty.append(chargeable_fees / self.capitals[-1]) 
            self.capitals[-1]-= chargeable_fees
        
            baseclose_prev = baseclose_row
            self.nominalss.append(nominal_tot)
            self.leverages.append(nominal_tot / self.capitals[-1])
            self.units_held.append(positions)
            self.targets_log.append(targets)
            self.inertia_log.append(inertias)
            self.weights_held.append(weights)            
            #end loop
        
        """
        capitals, capital ret, costs 123, leverage, strat scalar, weights, positions
        """ 
        units_df = pd.DataFrame(data=self.units_held, index=trade_datetime_range, columns=[inst + " units" for inst in self.instruments])
        targets_df = pd.DataFrame(data=self.targets_log, index=trade_datetime_range, columns=[inst + " targets" for inst in self.instruments])
        weights_df = pd.DataFrame(data=self.weights_held, index=trade_datetime_range, columns=[inst + " w" for inst in self.instruments])
        inertias_df = pd.DataFrame(data=self.inertia_log, index=trade_datetime_range, columns=[inst + " inertia" for inst in self.instruments])
       
        nominals_ser = pd.Series(data=self.nominalss, index=trade_datetime_range, name="nominal_tot")
        stratscal_ser = pd.Series(data=self.strat_scalars, index=trade_datetime_range, name="strat_scalar")
        leverages_ser = pd.Series(data=self.leverages, index=trade_datetime_range, name="leverages")
        
        execpen_ser = pd.Series(data=self.exec_penalty, index=trade_datetime_range, name="exec_penalty") 
        commpen_ser = pd.Series(data=self.comm_penalty, index=trade_datetime_range, name="comm_penalty")
        swappen_ser = pd.Series(data=self.swap_penalty, index=trade_datetime_range, name="swap_penalty") 
        chargeable_ser = pd.Series(data=self.chargeable_penalty, index=trade_datetime_range, name="cost_penalty")

        nominal_ret_ser = pd.Series(data=self.nominal_rets, index=trade_datetime_range, name="nominal_ret")
        capital_ret_ser = pd.Series(data=self.capital_rets, index=trade_datetime_range, name="capital_ret")
        capital_ser = pd.Series(data=self.capitals, index=trade_datetime_range, name="capital")  

        self.weights_df = weights_df.copy()
        self.positions_df = units_df.copy()
        self.targets_df = targets_df.copy()
        self.inertias_df = inertias_df.copy()
        
        self.capital_ser = capital_ser.copy()
        self.leverages_ser = leverages_ser.copy()            

        self.portfolio_df = pd.concat([
            units_df,
            weights_df,    
            stratscal_ser,
            nominals_ser,
            stratscal_ser,
            leverages_ser,
            execpen_ser,
            commpen_ser,
            swappen_ser,
            chargeable_ser,
            nominal_ret_ser,
            capital_ret_ser,
            capital_ser
        ], axis=1)
        
        if verbose:    
            print(self.portfolio_df)

        return self.portfolio_df

class Amalgapha(BaseAlpha):

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
        weightss=[], 
        leveragess=[],   
        strat_weights=None
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
        self.weightss = weightss
        self.leveragess = leveragess
        self.strat_weights = np.ones(len(self.weightss)) / len(self.weightss) \
            if not strat_weights else strat_weights

    def param_generator(self, shattered, param_idx=0):
        return super().param_generator(shattered=shattered)

    async def compute_metas(self, index, delta_lag, shattered, param_idx):
        await (super().compute_metas(index))
        df = pd.DataFrame(index=index)
        weights_dfs = []
        leverages_dfs = []
        for weights in self.weightss:
            weights_df = df.join(pd.DataFrame(weights))
            weights_dfs.append(weights_df)
        for leverages in self.leveragess:
            leverages_df = df.join(pd.DataFrame(leverages)) 
            leverages_dfs.append(leverages_df)
        
        self.weights_dfs = weights_dfs
        self.leverages_dfs = leverages_dfs
        pass
    
    def post_risk_management(self, nominal_tot, positions, weights, *args, **kwargs):
        return nominal_tot, positions, weights

    def set_positions(self, capital, strat_scalar, portfolio_vol, prev_positions, \
        ret_i=None, baseclose_row=None, **kwargs):
        
        leveraged_weights = np.array([
            weights_df.loc[ret_i].values * leverages_df.at[ret_i, "leverages"] \
            for weights_df, leverages_df in zip(self.weights_dfs, self.leverages_dfs)
        ])
        
        portfolio_weights = np.zeros(len(self.instruments))
        for i in range(len(self.strat_weights)):
            portfolio_weights += self.strat_weights[i] * leveraged_weights[i]

        targets = strat_scalar * capital * portfolio_weights / baseclose_row
        
        change = targets - prev_positions
        percent_change = np.abs(change) / np.abs(targets)
        inertia = self.positional_inertia
        inertia_override = 0.0 + (percent_change > inertia)
        not_inertia_override = 0.0 + (percent_change <= inertia)
        positions = inertia_override * targets + not_inertia_override * prev_positions

        positions = np.nan_to_num(positions, nan=0, posinf=0, neginf=0)
        targets = np.nan_to_num(targets, nan=0, posinf=0, neginf=0)
        nominal_tot = np.linalg.norm(positions * baseclose_row, ord=1)
        return positions, targets, nominal_tot, inertia_override
    
    def zip_data_generator(self):
        for (portfolio_i, portfolio_row),\
            (ret_i, ret_row),\
            (baseclose_i, baseclose_row), \
                in zip(\
            self.portfolio_df.iterrows(),\
            self.retdf.iterrows(),\
            self.baseclosedf.iterrows()
        ): 
            portfolio_row = portfolio_row.values.astype('float64')
            ret_row = ret_row.values.astype('float64')
            baseclose_row = baseclose_row.values.astype('float64')
            yield {
                "portfolio_i": portfolio_i, 
                "portfolio_row": portfolio_row, 
                "ret_i": ret_i, 
                "ret_row": ret_row, 
                "baseclose_i": baseclose_i, 
                "baseclose_row": baseclose_row
            }

class Alpha(BaseAlpha, ABC):
    
    @abstractmethod
    def compute_forecasts(self, portfolio_i, date, eligibles_row):
        pass    

    def post_risk_management(self, nominal_tot, positions, weights, date_idx=None, eligibles_row=None, *args, **kwargs):
        return nominal_tot, positions, weights

    def zip_data_generator(self):
        for (portfolio_i, portfolio_row),\
            (ret_i, ret_row),\
            (baseclose_i, baseclose_row), \
            (eligibles_i, eligibles_row),\
            (invrisk_i, invrisk_row),\
                in zip(\
            self.portfolio_df.iterrows(),\
            self.retdf.iterrows(),\
            self.baseclosedf.iterrows(),\
            self.eligiblesdf.iterrows(),\
            self.invriskdf.iterrows(),\
        ):  
            portfolio_row = portfolio_row.values.astype('float64')
            ret_row = ret_row.values.astype('float64')
            baseclose_row = baseclose_row.values.astype('float64')
            eligibles_row = eligibles_row.values.astype('int32')
            invrisk_row = invrisk_row.values.astype("float64")
            yield {
                "portfolio_i": portfolio_i, 
                "portfolio_row": portfolio_row, 
                "ret_i": ret_i, 
                "ret_row": ret_row, 
                "baseclose_i": baseclose_i, 
                "baseclose_row": baseclose_row, 
                "eligibles_i": eligibles_i,
                "eligibles_row": eligibles_row, 
                "invrisk_i": invrisk_i, 
                "invrisk_row": invrisk_row, 
            }
    
    def set_positions(self, capital, strat_scalar, portfolio_vol, prev_positions, \
        portfolio_i=None, ret_i=None, eligibles_row=None, invrisk_row=None, baseclose_row=None, **kwargs):
        
        forecasts, num_trading = self.compute_forecasts(portfolio_i=portfolio_i, date=ret_i, eligibles_row=eligibles_row)
        vol_target = 1 \
            / max(1, num_trading) \
            * capital \
            * portfolio_vol \
            / np.sqrt(253)
        
        targets = eligibles_row \
            * strat_scalar \
            * vol_target \
            * forecasts \
            * invrisk_row \
            / baseclose_row

        change = targets - prev_positions   
        percent_change = np.abs(change) / np.abs(targets)
        inertia = self.positional_inertia
        inertia_override = 0.0 + (percent_change > inertia)
        not_inertia_override = 0.0 + (percent_change <= inertia)
        positions = inertia_override * targets + not_inertia_override * prev_positions

        positions = np.nan_to_num(positions, nan=0, posinf=0, neginf=0)
        targets = np.nan_to_num(targets, nan=0, posinf=0, neginf=0)
        nominal_tot = np.linalg.norm(positions * baseclose_row, ord=1)
        return positions, targets, nominal_tot, inertia_override

