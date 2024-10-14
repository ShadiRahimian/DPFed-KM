import numpy as np
import pandas as pd
import preprocessing as pp
import helpers as h
from scipy.stats import bootstrap
from lifelines import KaplanMeierFitter
from lifelines import statistics as stat
from lifelines.utils import median_survival_times

def pvalue(df, dfs):
    """
    return p-value suitable to put on plots
    args:
        df: the original dataset to compare to 
        dfs: the secondary or reconstructed dataset
    """
    p = stat.logrank_test(df["duration"], dfs["duration"], df["event"], dfs["event"])
    p = np.round(p.p_value, 2)
    return p

def median(df):
    """
    calculate the median of the Kaplan-Meier estimator for df
    """
    kmf = KaplanMeierFitter()
    km = kmf.fit(df["duration"], df["event"])
    m = round(km.median_survival_time_)
    return m

def median_confidence(df):
    """
    calculate the median of the Kaplan-Meier estimator for df
    """
    kmf = KaplanMeierFitter()
    kmreal = kmf.fit(df["duration"], df["event"])
    median = round(kmreal.median_survival_time_)
    #95% confidence interval of the median
    median_ci = np.array(median_survival_times(kmreal.confidence_interval_).reset_index(level=0))
    return median, median_ci

def cmd_calculator(df, surr):
    """
    calculates the calibrated median difference
    
    cmd = |median_surr - median_df|/median_df
    """
    true_m = median(df)
    m = median(surr)
    cmd = np.round(abs(m - true_m) / true_m, 3)
    return cmd

def survival_rate_at_time(df, t_arr):
    """
    returns survival rates of the dataframe df at times in t_arr array 
    args:
        df: survival dataset Pandas dataframe
        t_arr: an array of times to calculate the survival function confidence
            for. These are real times
    returns:
        value_at_time: survival value at times t_arr
    """
    kmf = KaplanMeierFitter()
    kmfo = kmf.fit(df["duration"], df["event"])
    value_at_time = np.around(np.array(kmfo.survival_function_at_times(t_arr)), 2) #value of KM at tarr
    return value_at_time

def recons_ex(df, bins_length, max_time, n):
    """
    returns cmd and p-value for reconstruction experiments
    args:
        df: original pandas dataframe
        bins_length: discretization length for times
        max_time: maximum time considered for the study
        n: int number of points to calculate the surrogate dataset with
    """
    df_dig, bins = pp.digitize_time(df, bins_length, max_time)
    s = h.df2s(df_dig, bins)
    y = h.s2y(s)
    surr = h.surrogates(bins, y, n)
    p = pvalue(df, surr)
    cmd =cmd_calculator(df, surr)

    return p, cmd

def centralized_ex(df, surr, t_arr):
    """
    returns parameters needed for centralized experiments
    args:
        df: original survival dataframe
        surr: reconstructed surrogate dataframe after method
        t_arr: an array of times to calculate the survival function confidence
            for. These are real times
    returns:
        p: p-value between df and surr
        m: median of surr dataset
        sr: survival rates of surr at times of t_arr
    """
    p = pvalue(df, surr)
    m = median(surr)
    sr = survival_rate_at_time(surr, t_arr)
    return p, m , sr

def confidence_of_mean(samples, n_res=9999, cl=0.95, r_state=1234):
    """returns vectors of confidence for mean of samples"""
    sample_mean = np.around(np.mean(samples, axis=0), 2)
    samples = (samples,)
    res = bootstrap(samples, np.mean, n_resamples=n_res, confidence_level=cl, method="basic", random_state=r_state)
    return sample_mean, np.around(res.confidence_interval, 2)

def hyperparameter_ex(df, surr):
    """
    returns parameters needed for hyperparameter tuning experiments
    args:
        df: original survival dataframe
        surr: reconstructed surrogate dataframe after method
    returns:
        p: p-value between df and surr
        cmd: calibrated median difference between df and surr
    """
    p = pvalue(df, surr)
    cmd = cmd_calculator(df, surr)
    return p, cmd
    
def confidence_at_time(df, t_arr):
    """
    calculates confidence for any time in the valid timeframe

    args:
        df: survival dataset Pandas dataframe
        t_arr: an array of times to calculate the survival function confidence
            for. These are real times
    returns:
        value_at_time: survival value at times t_arr
        upper: upper KM_estimate_0.95 at times t_arr
        lower: lower KM_estimate_0.95 at times t_arr
    """
    kmf = KaplanMeierFitter()
    kmfo = kmf.fit(df["duration"], df["event"])
    value_at_time = np.around(np.array(kmfo.survival_function_at_times(t_arr)), 2) #value of KM at tarr

    ci = kmfo.confidence_interval_.reset_index(level=0)
    upper = []
    lower = []
    for t in t_arr:
        cit = ci[ci["index"]<=t].iloc[-1:]
        u = np.around(cit.iloc[:, -1].values[0], 2)
        upper += [u]
        l = np.around(cit.iloc[:, -2].values[0], 2)
        lower += [l]

    return value_at_time, upper, lower

def ci_calculator(df):
    """
    Function to return higher order metrics over dataframe df
    """

    times = np.array(df["duration"])
    events = np.array(df["event"])

    kmf = KaplanMeierFitter()
    # fits a Kaplan Meier estimator to data
    kmf.fit(times, event_observed=events)

    sf = kmf.survival_function_
    sf = sf.reset_index(level=0)
    sf = sf.reset_index(level=0)

    median = kmf.median_survival_time_
    median_ci = median_survival_times(kmf.confidence_interval_)
    median_ci = median_ci.reset_index(level=0)

    dfci = kmf.confidence_interval_
    dfci = dfci.reset_index(level=0)

    return sf, median, median_ci, dfci


def run_metrics(df, time_points, bins_length=6, max_time=350):
    """
    Calculates the metrics for dataframe df
    
    args:
        df: dataframe to calculate the metrics on
        time_points: array of ints
            time points at which we wish to return the value of KM estiamtor
            and its corresponding confidence intervals
        bins_length: int, real time in the dataframe we wish to digitize to
        max_time: int, maximum time of the study to be considered for df

    returns:
        sf_times: pandas dataframe of KM values at times of time_points
        median: float, real median time of the dataset
        dfci: pandas dataframe of the the confidence intervals of the KM
            values at times of time_points. The first row returns the CI for
            the median time of the df
    """
    df_dig, bins = pp.digitize_time(df, bins_length, max_time)
    # sf, median, median_ci, ci = ci_calculator(df_dig)
    sf, median, median_ci, ci = ci_calculator(df)

    sf_times = pd.DataFrame([])

    dfci = pd.DataFrame([])
    dfci = pd.concat([dfci, median_ci])

    for t in time_points:
        sf_times = pd.concat([sf_times, sf[sf["timeline"] == t]])
        dfci = pd.concat([dfci, ci[ci["index"] == t]])

    dfci.insert(1, "KM_estimate", sf_times["KM_estimate"])

    return median, dfci

