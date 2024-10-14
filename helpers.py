import numpy as np
import pandas as pd
import scipy.stats as ss
import preprocessing as pp
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times


# returns True if the name holds
def strictly_increasing(L):
  return all(x < y for x, y in zip(L, L[1:]))


def strictly_decreasing(L):
    return all(x > y for x, y in zip(L, L[1:]))


def non_increasing(L):
    return all(x >= y for x, y in zip(L, L[1:]))


def non_decreasing(L):
    return all(x <= y for x, y in zip(L, L[1:]))


def make_non_increasing(L):
    for i in range(1, len(L)):
        L[i] = np.clip(L[i], a_min=0, a_max=L[i - 1])
    return L

def bins_vector(bin_length, max_time):

    nb = max_time//bin_length
    max_time = nb*bin_length

    bins = np.arange(0, max_time, bin_length)
    return bins
    
    
def event_number_counter(df):
    """
    gives count numbers for events 0, 1 and total number
    
    arg:
        df: Pandas dataFrame of survival data

    returns:
        ntot: int, total number of points in dataset
        n1: int, number of points with event 1
        n0: int number of censored points, note: n1+n0=ntot
    """

    ntot = len(df)
    n1 = len(df[df["event"]==1])
    n0 = len(df[df["event"]==0])

    if n0+n1 != ntot:
        raise Exception("n1+n0 =/= ntot")

    return ntot, n1, n0


def event_seperator(df):
    """
    Receives dataframe and returns two dataframes for event of 1 and 0

    args:
        df: Pandas dataFrame of survival data

    returns:
        df0: Pandas dataFrame with event=0
        df1: Pandas dataframe with event=1
    """

    df0 = df[df["event"] == 0].copy(deep=True)
    df1 = df[df["event"] == 1].copy(deep=True)

    return df0, df1


#Necessary functions for DP-Matrix #############################################

def data2event_counter(df):
    """ 
    receives dataframe of durations and converts to survival table

    args:
        df: pandas dataframe containing durations (for each data point). Note
            that preprocessing.digitzie_times is recommended to be used prior
            on the raw dataset.
    returns:
        survival_table: a Pandas dataFrame for the survival table
            for each rounded "duration" the number of observed events
            are reported. For each distinct time "duration", total number
            of incidents, "num_obs", number of events of 1, "event1", and 
            number of censored data, "event0", are reported. Note that 
            event0 + event1 = num_obs
        num_subjects: int, total number of subjects in the study
    """

    # first round up durations numbers to be able to better construct a
    # survival table:
    # df["duration"] = np.ceil(df["duration"]).astype(int)
    # ! use preprocessing.digitize_time() instead

    grp = df.groupby("duration")

    survival_table = pd.DataFrame({"num_obs": grp.size(),
                                   "event1": grp["event"].sum(),
                                   "event0": grp.size() - grp["event"].sum()})

    # add "duration" as a column to the dataset
    survival_table = survival_table.reset_index(level=0)

    num_subjects = survival_table["num_obs"].sum()

    return survival_table, num_subjects


def risk_counter(survival_table, num_subjects=0):
    """
    Calculates the number at risk at each time in table

    args:
        survival_table: pandas DataFrame containing the discrete "duration" 
        of events, "event1" and "event0", counting the number of these 
        types of events for each time in "duration.
        num_subjects: int, the total number of subjects in study

    returns:
        survival_table: same as input, appends a "risk" column which counts
        the risk set at each time of "duration"
    """
    if num_subjects == 0:
        num_subjects = (survival_table["event1"].sum() +
                        survival_table["event0"].sum())

    prior_count = (survival_table["event1"].cumsum() +
                   survival_table["event0"].cumsum()).shift(1, fill_value=0)

    survival_table.insert(0, "risk", num_subjects - prior_count)

    # so that the value of at risk does not go to negatives
    neg_idx = survival_table.index[survival_table["risk"] < 0]
    survival_table.drop(neg_idx, axis=0, inplace=True)

    # so that every every subject experiences at least one type of event during
    # the time of study
    idx = survival_table.tail(1).index  # index of last row
    remaining = (survival_table["risk"][idx] -
                 survival_table["event0"][idx] - survival_table["event1"][idx])
    if remaining.item() < 0:
        survival_table.loc[idx, "event1"] = 0
        survival_table.loc[idx, "event0"] = survival_table["risk"][idx]
    else:
        survival_table.loc[idx, "event0"] = survival_table["event0"][idx] + remaining.item()

    return survival_table


def count2duration(survival_table):
    """
    Converts survival table to df of duration/event
    Note: should pass output of risk_counter to.

    args:
        survival_table: pandas dataframe of survival counts

    returns:
        full_table: pandas dataframe of "duration" and "event" type of either 0
        or 1 for each data point
    """
    event0_part = survival_table.loc[
        survival_table.index.repeat(survival_table.event0)]
    event0_part["event"] = 0
    event1_part = survival_table.loc[
        survival_table.index.repeat(survival_table.event1)]
    event1_part["event"] = 1

    frames = [event0_part, event1_part]
    full_table = pd.concat(frames)
    full_table.drop(["event0", "event1", "risk"], axis=1, inplace=True)

    return full_table


#############################################################################
# functions necessary to convert dataset to S, S to y and vice versa

def y2s(probs):
    """
    takes probabilities vector y(t) and calculates KM values S(t)

    args:
        probs: array of probabilities y(t) which has one element more than S(t)
    returns:
        surv: array of corresponding S(t) values
    """
    surv = 1 - probs.cumsum()
    surv = surv[:-1]
    return surv


def s2y(survival_values):
    """
    takes array of discretized survival values and returns y (probabilities)

    args:
        surviaval_values: array in shape of bins. Survival values at the 
            value of each corresponding bin
    returns:
        probs: y vector of probabilities of incident. It has 1 element more 
            than survival_values to account for probability of dying after 
            max_time
    """

    survival_values = np.append(survival_values, 0.0)

    predscum = 1 - survival_values
    probs = np.insert(np.diff(predscum), 0, predscum[0])
    if not probs.sum() == 1.0:
        raise "probs should sum up to 1"

    return probs


def df2s(df, bins):
    """
    Function to convert df to S(t). 

    Gets pandas DataFrame of "duration" and "event" for each data point, 
    first calculates the KM estimator 

    args:
        df: pandas dataframe containing "duration" and "event" for each data
            point
        bins: np.array of interval limits for time digitization

    returns:
        survival_values: array of S(t) for value of Kaplan-Meier estimator 
            at each time bin
    """
    times = np.array(df["duration"])
    events = np.array(df["event"])

    kmf = KaplanMeierFitter()
    # fits a Kaplan Meier estimator to data
    kmf.fit(times,
            event_observed=events)
    # returns value of calculated Surv(t) at times bins
    survival_values = np.array(kmf.survival_function_at_times(bins))

    return survival_values


def surrogates(bins, probs, total_num):
    """
    builds surrogate time and event vector for y(t) values, so the 
    log-rank test can be applied

    Args
        bins: array of ints, returned by digitizing the times
        probs: array noisy probs
        total_num: int
            total number of data points 
    """
    surrogate_times = []
    surrogate_events = []
    num_per_time = np.around(probs * total_num)
    for i in np.arange(bins.shape[0]):
        num_events = np.full((num_per_time[i].astype(int)), bins[i])
        surrogate_times = np.append(surrogate_times, num_events)
    surrogate_events = np.ones(surrogate_times.shape)

    # account for the last element of y(t) which is outside of max_time
    num_events = np.full((num_per_time[-1].astype(int)), bins[-1])
    surrogate_times = np.append(surrogate_times, num_events)
    surrogate_events = np.append(surrogate_events, np.zeros(num_events.shape))

    surrogate_df = pd.DataFrame({"duration": surrogate_times,
                                 "event": surrogate_events})

    return surrogate_df

#############################################################################
