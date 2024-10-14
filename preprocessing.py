import numpy as np
import pandas as pd


def digitize_time(df, bins_length, max_time=0):
    """digitizes times in times

    e.g: bin_length = 4 months, max_time=12 ->
    0    4    8    12
     |----||---||-------------

    args:
        bins_length: int
            the length of time bins in real time, e.g. months 
        df: pandas dataframe containing "duration" and "event" for each data
            point
        max_time : float
            maximum real time that is desired to be considered for dataset
            
    returns:
        bins: an array of bin intervals
        df_dig: same as df but with binned times of event, if points fall after 
            max_time, their event will be zero (censored) 
    """
    times = np.array(df["duration"])
    events = np.array(df["event"])

    if max_time==0:
        nb = times.max()//bins_length
        max_time = nb*bins_length #sets max_time to rounded down multiple
    else: 
        nb = max_time//bins_length
        max_time = nb*bins_length

    ind = np.where(times>max_time)
    events[ind]=0

    bins = np.arange(0, max_time, bins_length)
    times_digitized = np.digitize(times, bins, right=True)*bins_length
    bins = np.append(bins, max_time) #will be used later to convert S(t) to y

    df_dig = pd.DataFrame({"duration": times_digitized, "event": events})

    return df_dig, bins
