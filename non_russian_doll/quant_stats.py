import math
import random
import scipy
import asyncio
import numpy as np
import pandas as pd
import mplfinance as mpf
import multiprocess as mp
import statsmodels.api as sm
import matplotlib.pyplot as plt

from scipy import stats
from copy import deepcopy
from functools import partial
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathos.multiprocessing import ProcessingPool

# from quantyhlib.general_utils import timeme

'''Utility Functions'''
def adjust_prices(dfs):
    ohlc = ["open", "high", "low", "close"]
    assert type(dfs) == type(list()) or type(dfs) == type(dict())
    if type(dfs) == type(list()):
        res = []
        for df in dfs:
            if "adj_close" in df:
                price_adjustment = df["adj_close"] / df["close"]
                df[ohlc] = df[ohlc].mul(price_adjustment, axis=0)
                res.append(df)
            else:
                res.append(df)
        return res
    elif type(dfs) == type(dict()):
        res = {}
        for k,v in dfs.items():
            if "adj_close" in v:
                price_adjustment = v["adj_close"] / v["close"]
                v[ohlc] = v[ohlc].mul(price_adjustment, axis=0)
                res[k] = v
            else:
                res[k] = v
        return res

'''Permutation and Hypothesis Testing'''

'''
Permute Methods
'''
#NOTE 16
def generate_permutations(array):
    if len(array) <= 1:
        return [array]
    permutations = []
    for i in range(len(array)):
        element = array[i]
        sub_sequence = array[:i] + array[i+1:]
        sub_permutations = generate_permutations(sub_sequence)
        for permutation in sub_permutations:
            permutations.append([element] + permutation)
    assert len(permutations) == np.math.factorial(len(array))
    return permutations

#NOTE 17
def permutation_member(array):
    i = len(array)
    while i > 1:
        j = int(np.random.uniform(0, 1) * i) 
        if j >= i : j = i - 1
        i -= 1
        temp = array[i]
        array[i] = array[j]
        array[j] = temp
    return array

def permute_price(price, permute_index=None):
    if not permute_index: 
        permute_index = permutation_member(list(range(len(price) - 1)))
    log_prices = np.log(price)
    diff_logs = log_prices[1:] - log_prices[:-1]
    diff_perm = diff_logs[permute_index]
    cum_change = np.cumsum(diff_perm)
    new_log_prices = np.concatenate(([log_prices[0]], log_prices[0] + cum_change))
    new_prices = np.exp(new_log_prices)
    return new_prices

def permute_multi_prices(prices):
    assert all([len(price) == len(prices[0]) for price in prices]) 
    permute_index = permutation_member(list(range(len(prices[0]) - 1)))
    new_prices = [permute_price(price, permute_index=permute_index) for price in prices]
    return new_prices

#NOTE 18
def permute_bars(ohlcv, index_inter_bar=None, index_intra_bar=None):
    if len(ohlcv) <= 2:
        return deepcopy(ohlcv[["open", "high", "low", "close", "volume"]])
    if not index_inter_bar: 
        index_inter_bar = permutation_member(list(range(len(ohlcv) - 1)))
    if not index_intra_bar:
        index_intra_bar = permutation_member(list(range(len(ohlcv) - 2))) 
    log_data = np.log(ohlcv.astype("float32"))
    delta_h = log_data["high"].values - log_data["open"].values
    delta_l = log_data["low"].values - log_data["open"].values
    delta_c = log_data["close"].values - log_data["open"].values
    diff_deltas_h = np.concatenate((delta_h[1:-1][index_intra_bar], [delta_h[-1]]))
    diff_deltas_l = np.concatenate((delta_l[1:-1][index_intra_bar], [delta_l[-1]])) 
    diff_deltas_c = np.concatenate((delta_c[1:-1][index_intra_bar], [delta_c[-1]]))

    new_volumes = np.concatenate(
        (
            [log_data["volume"].values[0]], 
            log_data["volume"].values[1:-1][index_intra_bar], 
            [log_data["volume"].values[-1]]
        )
    )
    
    inter_open_to_close = log_data["open"].values[1:] - log_data["close"].values[:-1]
    diff_inter_open_to_close = inter_open_to_close[index_inter_bar]
     
    new_opens, new_highs, new_lows, new_closes = \
        [log_data["open"].values[0]], \
        [log_data["high"].values[0]], \
        [log_data["low"].values[0]], \
        [log_data["close"].values[0]]

    last_close = new_closes[0]
    for i_delta_h, i_delta_l, i_delta_c, inter_otc in zip(
        diff_deltas_h, diff_deltas_l, diff_deltas_c, diff_inter_open_to_close
    ):
        new_open = last_close + inter_otc
        new_high = new_open + i_delta_h
        new_low = new_open + i_delta_l
        new_close = new_open + i_delta_c
        new_opens.append(new_open)
        new_highs.append(new_high)
        new_lows.append(new_low)
        new_closes.append(new_close)
        last_close = new_close

    new_df = pd.DataFrame(
        {
            "open": new_opens,
            "high": new_highs,
            "low": new_lows,
            "close": new_closes,
            "volume": new_volumes
        }
    )
    new_df = np.exp(new_df)
    new_df.index = ohlcv.index
    return new_df

#NOTE 19
def permute_multi_bars(bars):
    if all([len(bar) == len(bars[0]) for bar in bars]):
        index_inter_bar = permutation_member(list(range(len(bars[0]) - 1)))
        index_intra_bar = permutation_member(list(range(len(bars[0]) - 2)))
        new_bars = [
            permute_bars(
                bar, 
                index_inter_bar=index_inter_bar, 
                index_intra_bar=index_intra_bar
            ) 
            for bar in bars
        ]
    else:
        bar_indices = list(range(len(bars)))
        index_to_dates = {k:set(list(bar.index)) for k,bar in zip(bar_indices, bars)}
        date_pool = set()
        for index in list(index_to_dates.values()):
            date_pool = date_pool.union(index)
        date_pool = list(date_pool)
        date_pool.sort()
        partitions, partition_idxs = [], []
        temp_partition = []
        temp_set = set([idx for idx,date_sets in index_to_dates.items() \
            if date_pool[0] in date_sets])
        
        for i_date in date_pool:
            i_insts = set()
            for inst, date_sets in index_to_dates.items():
                if i_date in date_sets:
                    i_insts.add(inst)
            if i_insts == temp_set:
                temp_partition.append(i_date)
            else:
                partitions.append(temp_partition)
                partition_idxs.append(list(temp_set))
                temp_partition = [i_date]
                temp_set = i_insts
        partitions.append(temp_partition)
        partition_idxs.append(list(temp_set))

        chunked_bars = defaultdict(list)
        for partition, idx_list in zip(partitions, partition_idxs):
            permuted_bars = permute_multi_bars(
                [bars[idx].loc[partition] for idx in idx_list]
            )
            for idx, bar in zip(idx_list, permuted_bars): 
                chunked_bars[idx].append(bar)
       
        new_bars = [None] * len(bars)
        for idx in bar_indices:
            new_bars[idx] = pd.concat(chunked_bars[idx], axis=0)
    return new_bars

'''
Permutation Tests
'''
'''Hypothesis Testing for k marginal signals with strong control for familywise error rate'''
#NOTE 20
async def marginal_family_test(
    criterion_function, alpha_family, member_stats_generator=None, m=1000, alpha=0.15
):
    if not member_stats_generator:
        async def get_member_stats(member):
            await member.run_simulation()
            zero_filtered_stats = member.get_zero_filtered_stats()
            retdf, leverages, weights = \
                zero_filtered_stats["retdf"], \
                zero_filtered_stats["leverages"], \
                zero_filtered_stats["weights"]
            return {"rets": retdf, "leverages": leverages, "weights" : weights} 
        member_stats_generator = get_member_stats
    unpermuted_stats = await asyncio.gather(
        *[member_stats_generator(member) for member in alpha_family]
    )
    unpermuted_criterions = np.array([criterion_function(**stat) for stat in unpermuted_stats])

    round_criterions = []
    for _ in range(m):
        family_datacopies = [adjust_prices
            (deepcopy(member.datacopy))
            for member in alpha_family
        ]
        family_insts = np.array([list(family_dc) for family_dc in family_datacopies], dtype="object")
        family_sizes = np.array([len(family_inst) for family_inst in family_insts])
        family_insts = family_insts.flatten()
        shuffled_bars = permute_multi_bars(
            np.array([
                list(family_datacopy.values()) 
                for family_datacopy in family_datacopies
            ], dtype="object").flatten(order="C")
        )#NOTE 21

        shuffled_family_copies = []
        for start, end in zip(
            np.concatenate(([0], np.cumsum(family_sizes)[:-1])), 
            np.cumsum(family_sizes)
        ):
            shuffled_member_copy = {
                k:v for k,v in zip(family_insts[start:end], shuffled_bars[start:end])
            }
            shuffled_family_copies.append(shuffled_member_copy)
        
        alpha_family_copy = deepcopy(alpha_family)
        for member_copy, shuffled_copy in zip(alpha_family_copy, shuffled_family_copies):
            member_copy.datacopy = shuffled_copy
 
        round_stats = await asyncio.gather(
            *[member_stats_generator(member_copy) for member_copy in alpha_family_copy]
        )
        round_criterions.append(
             np.array([criterion_function(**stat) for stat in round_stats])
        )
    return stepdown_algorithm(unpermuted_criterions, round_criterions, alpha)

#NOTE 22
def stepdown_algorithm(unpermuted_criterions, round_criterions, alpha):
    pvalues = np.array([None] * len(unpermuted_criterions))
    exact = np.array([False] * len(unpermuted_criterions), dtype=bool)
    indices = np.array(list(range(len(unpermuted_criterions))))
    while not all(exact):
        stepwise_indices = indices[~exact]
        stepwise_criterions = np.array(unpermuted_criterions)[stepwise_indices]
        member_count = np.zeros(len(stepwise_criterions))
        for round in range(len(round_criterions)):
            round_max = np.max(np.array(round_criterions[round])[stepwise_indices])
            member_count += (0.0 + round_max >= np.array(stepwise_criterions))
        bounded_pvals = (1 + member_count) / (len(round_criterions) + 1)
        if np.min(bounded_pvals) < alpha:
            best_member = np.argmin(bounded_pvals)
            pvalues[stepwise_indices[best_member]] = np.min(bounded_pvals)
            exact[stepwise_indices[best_member]] = True
        else:
            for bounded_p, index in zip(bounded_pvals, stepwise_indices):
                pvalues[index] = bounded_p
            break
    return pvalues, exact

'''Generic Code for Decision Shufflers and Data Shuffler test, 
timers_pvalue, pickers_pvalue, trader_pvalue'''
#NOTE 23
def permute_in_new_event_loop(criterion_function, generator_function, size, kwargs):
    async def batch_shuffler(size):
        kwargss = await asyncio.gather(
            *[generator_function(**kwargs) for _ in range(size)]
        )
        criterions = [criterion_function(**kwargs) for kwargs in kwargss]
        return criterions
    return asyncio.run(batch_shuffler(size))

def split_rounds(m, max_splits=mp.cpu_count() * 2):
    batch_size = math.ceil(m / max_splits)
    batch_sizes = []
    while m > 0:
        batch_sizes.append(min(batch_size, m))
        m -= batch_size
    return batch_sizes

#NOTE 25
def init_pools():
    random.seed()    

#NOTE 24
async def permutation_shuffler_test(criterion_function, generator_function, m=1000, **kwargs):
    print(f"permuation test in progress for {generator_function} {m} times...")
    unpermuted = criterion_function(**kwargs)
    batch_sizes = split_rounds(m=m)
    f = lambda args: permute_in_new_event_loop(*args)
    with ProcessingPool(initializer=init_pools) as pool:
        batched_criterions = pool.map(f, [
            (criterion_function, generator_function, size, kwargs) for size in batch_sizes
        ])
    criterions = []
    for batched_criterion in batched_criterions:
        criterions.extend(batched_criterion)
    p = (1 + np.sum(criterions >= unpermuted)) / (len(criterions) + 1)
    return p

'''
Hypothesis Tests
'''
def one_sample_wilcoxon_signed_rank(sample, m0, side="greater"):
    n = len(sample)
    ranks = stats.rankdata(np.abs(sample - m0))
    signs = np.sign(sample - m0)
    signed_ranks = signs * ranks
    w = np.sum(signed_ranks[signed_ranks > 0])
    ew = n * (n + 1) / 4
    varw = n * (n + 1) * (2 * n + 1) / 24
    z = (w - ew - 0.5) / np.sqrt(varw) \
        if side == "greater" else (w - ew + 0.5) / np.sqrt(varw)
    p = 1 - stats.norm.cdf(z) if side == "greater" else stats.norm.cdf(z)
    #stats.wilcoxon(sample - m0, alternative=side)
    return p

def one_sample_sign_test(sample, m0, side="greater", norm_approx=False):
    assert side == "greater" or side == "lesser"
    n = len(sample)
    S = np.sum((sample - m0) > 0) if side == "greater" else np.sum((sample - m0) < 0)
    'S ~ binom(n, 0.5)'
    if norm_approx: #(w cc)
        if side == "greater":
            Z = ( S - n * 0.5 - 0.5) / np.sqrt( n / 4) 
            p = 1 - stats.norm.cdf(Z)
            return p
        else:
            Z = (S - n * 0.5 + 0.5) / np.sqrt( n / 4)
            p = stats.norm.cdf(Z)
            return p 
    return stats.binom_test(S, n=n, p=0.5, alternative=side)

def one_sample_bootstrap_test(sample, lambda_estimator, null_value=0, side="greater", N=1000):
    estimate = lambda_estimator(sample)
    bootstrap_samples = [random.choices(sample, k=len(sample)) for _ in range(N)]
    bootstrap_sample_estimates = [lambda_estimator(bootstrap_sample) for bootstrap_sample in bootstrap_samples]
    bootstrap_standard_error = np.std(bootstrap_sample_estimates)
    Z = (estimate - null_value) / bootstrap_standard_error
    return 1 - stats.norm.cdf(Z) if side == "greater" else stats.norm.cdf(Z)

def one_sample_t_test(sample, mu0, side="greater"):
    res = stats.ttest_1samp(sample, mu0, alternative=side)
    return res.pvalue, res.statistic

'''Others'''
def partition_df(df, size):
    partitions = []
    while len(df) != 0:
        partitions.append(df.head(size))
        df = df.tail(len(df) - size)
    return partitions

def pick_orthogonal_candidate(xs, ys):
    rsquared = []
    for y in ys:
        aligner = pd.DataFrame(index=y.index)
        temp = [aligner.join(x).reset_index(drop=True) for x in xs]
        df=pd.concat([y.reset_index(drop=True), *temp], axis=1)
        df.columns = ["y", *[f"x{i}" for i in range(len(list(df)) - 1)]]
        df = df.dropna(axis=0)
        y = df.y
        x = df[list(df)[1:]]
        x = sm.add_constant(x)
        model = sm.OLS(y,x)
        result = model.fit()
        rsquared.append(result.rsquared)    
    return np.argmin(rsquared)

def piecewise_principal_component(seriess, date_idx, piecelen):
    aligner = pd.DataFrame(index=date_idx)
    aligned_series = []
    for series in seriess:
        aligned_series.append(series)
    aligned_df = pd.concat(aligned_series, axis=1)
    aligned_df.columns = list(range(len(seriess)))
    partitions = partition_df(df=aligned_df, size=piecelen)
    principal_components = []
    for partition in partitions:
        partition = partition.dropna(axis=1,thresh=int(0.20 * len(partition)))
        partition = partition.dropna(axis=0, how="all")
        partition = partition.T.fillna(partition.mean(axis=1)).T
        z1, *res = get_principal_components(X=partition, components=1)
        principal_components.append(z1)
    piecewise_principal_component = pd.concat(principal_components, axis=0)
    piecewise_principal_component[0] = np.real(piecewise_principal_component.values)
    return aligner.join(piecewise_principal_component)

def get_principal_components(X, components=None):
    if not components: components = len(X)
    X = X - X.mean()
    Z = X / X.std()
    S = np.dot(Z.T, Z)
    eigenvalues, eigenvectors = np.linalg.eig(S)
    prop_var = [eigenvalue / np.sum(eigenvalues) for eigenvalue in eigenvalues]
    cum_var = np.cumsum(prop_var)
    proj_matrix = eigenvectors.T[:][:components].T
    principal_components = X.dot(proj_matrix)
    return principal_components, cum_var