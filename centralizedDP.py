import numpy as np
import pandas as pd
import math
import scipy.fft as fft
from scipy.special import softmax
from scipy import stats
from sklearn.isotonic import IsotonicRegression

import helpers as h
import preprocessing as pp

#to convert y(t) to S(t) we have:
    # surv = 1 - probs.cumsum()

# to plot the S(t) that is calculated from probs y(t):
# plt.step(bins, surv[:-1], where="post")

class CentralizedDPS():
    """
        args:
            df: pandas dataframe containing "duration" and "event" for each data
                point
            timestep (bins_length for preprocessing):int
                the length of time bins in real time, e.g. months
            max_time: int
                maximum time scale to be considered for the dataset
            epsilon: float
                privacy budget for DP with Laplace noise
            frac: float
                fraction of first coefficients of DCT to take out of all
            num: int
                If negative it means the sensitivity would be calculated 
                according to the df that is passed as the argument. Otherwise, we
                are in a collaborative setting, and the sensitivity is based on
                the smallest dataset size with size num.
            k: int, the number of first coefficients to take for compression

    """
    def __init__(self, df, bin_length, max_time, frac, epsilon):
        self.df = df
        self.bin_length = bin_length
        self.max_time = max_time
        self.frac = frac
        self.epsilon = epsilon

    def DP_Surv(self):
        """
        DP compression of signal with Discrete Cosine Transform
        """
        df_dig, bins = pp.digitize_time(self.df, self.bin_length, self.max_time)
        num_bins = np.shape(bins)[0]
        #sample the S(t) function at bins[i]
        values = h.df2s(df_dig, bins)
        
        #number of uncensored data points and total number of data 
        ntot, n1, n0 = h.event_number_counter(self.df)

        #make a mask that zeros the unwanted coefficients
        # 1 1 1 0 0 0 0 0 0 0 
        #|-k---||num_bins-k--|
        k = round(self.frac*num_bins)
        mask = np.zeros(values.shape)
        mask[:k] = 1

        # l2 = ((1+n0)/ntot) * np.sqrt(num_bins -1)
        l2 = (1/ntot) * np.sqrt(num_bins -1)
        sensitivity = np.sqrt(k) * l2 #l1 sensitivity of S(t)

        dct = fft.dct(values, norm="ortho")

        noise = np.random.laplace(0, sensitivity/self.epsilon, size=dct.size)
        noisy_dct = dct + noise

        compressed_noisy_surv = fft.idct(mask * noisy_dct, norm="ortho")
        compressed_noisy_surv[0] = 1.0
        #make the signal non-increasing and between limits
        ir = IsotonicRegression(increasing=False, y_min=0.0, y_max=1.0, out_of_bounds='clip')
        compressed_noisy_surv = ir.fit_transform(bins, compressed_noisy_surv)

        if h.non_increasing(compressed_noisy_surv) == False:
            raise Exception("post-process the noisy surv from DP-Surv to make it non-increasing")

        # # compressed_noisy_surv made between 0 and 1
        # compressed_noisy_surv = np.clip(compressed_noisy_surv, 0, 1.0)
        # # compressed noisy surv made non-increasing
        # compressed_noisy_surv = h.make_non_increasing(compressed_noisy_surv)

        noisy_probs = h.s2y(compressed_noisy_surv)
        surr_dataset = h.surrogates(bins, noisy_probs, ntot)

        return surr_dataset, noisy_probs, compressed_noisy_surv, bins

    def DP_Surv_w(self):
        """
        DP compression of signal with Discrete Cosine Transform
        """
        df_dig, bins = pp.digitize_time(self.df, self.bin_length, self.max_time)
        num_bins = np.shape(bins)[0]
        #sample the S(t) function at bins[i]
        values = h.df2s(df_dig, bins)
        
        #number of uncensored data points and total number of data 
        ntot, n1, n0 = h.event_number_counter(self.df)

        #make a mask that zeros the unwanted coefficients
        # 1 1 1 0 0 0 0 0 0 0 
        #|-k---||num_bins-k--|
        k = round(self.frac*num_bins)
        mask = np.zeros(values.shape)
        mask[:k] = 1

        l2 = (1/ntot) * np.sqrt(num_bins) * (1+n0)
        delta = 0.001
        alpha = self.epsilon/2
        beta = self.epsilon / (4*k + 4*np.log(2/delta))
        sensitivity = np.exp(-beta)* np.sqrt(k) * l2 #l1 sensitivity of S(t)

        dct = fft.dct(values, norm="ortho")

        noise = (sensitivity/alpha) * np.random.laplace(0, 1, size=dct.size)
        noisy_dct = dct + noise

        compressed_noisy_surv = fft.idct(mask * noisy_dct, norm="ortho")
        compressed_noisy_surv[0] = 1.0
        #make the signal non-increasing and between limits
        ir = IsotonicRegression(increasing=False, y_min=0.0, y_max=1.0, out_of_bounds='clip')
        compressed_noisy_surv = ir.fit_transform(bins, compressed_noisy_surv)

        if h.non_increasing(compressed_noisy_surv) == False:
            raise Exception("post-process the noisy surv from DP-Surv to make it non-increasing")

        # # compressed_noisy_surv made between 0 and 1
        # compressed_noisy_surv = np.clip(compressed_noisy_surv, 0, 1.0)
        # # compressed noisy surv made non-increasing
        # compressed_noisy_surv = h.make_non_increasing(compressed_noisy_surv)

        noisy_probs = h.s2y(compressed_noisy_surv)
        surr_dataset = h.surrogates(bins, noisy_probs, ntot)

        return surr_dataset, noisy_probs, compressed_noisy_surv, bins

    def DP_Surv_worst(self):
        """
        DP compression of signal with Discrete Cosine Transform
        """
        df_dig, bins = pp.digitize_time(self.df, self.bin_length, self.max_time)
        num_bins = np.shape(bins)[0]
        #sample the S(t) function at bins[i]
        values = h.df2s(df_dig, bins)
        
        #number of uncensored data points and total number of data 
        ntot, n1, n0 = h.event_number_counter(self.df)

        #make a mask that zeros the unwanted coefficients
        # 1 1 1 0 0 0 0 0 0 0 
        #|-k---||num_bins-k--|
        k = round(self.frac*num_bins)
        mask = np.zeros(values.shape)
        mask[:k] = 1

        # l2 = ((1+n0)/ntot) * np.sqrt(num_bins -1)
        l2 = (1/ntot) * np.sqrt(num_bins) *(1+ntot)
        sensitivity = np.sqrt(k) * l2 #l1 sensitivity of S(t)

        dct = fft.dct(values, norm="ortho")

        noise = np.random.laplace(0, sensitivity/self.epsilon, size=dct.size)
        noisy_dct = dct + noise

        compressed_noisy_surv = fft.idct(mask * noisy_dct, norm="ortho")
        compressed_noisy_surv[0] = 1.0
        #make the signal non-increasing and between limits
        ir = IsotonicRegression(increasing=False, y_min=0.0, y_max=1.0, out_of_bounds='clip')
        compressed_noisy_surv = ir.fit_transform(bins, compressed_noisy_surv)

        if h.non_increasing(compressed_noisy_surv) == False:
            raise Exception("post-process the noisy surv from DP-Surv to make it non-increasing")

        # # compressed_noisy_surv made between 0 and 1
        # compressed_noisy_surv = np.clip(compressed_noisy_surv, 0, 1.0)
        # # compressed noisy surv made non-increasing
        # compressed_noisy_surv = h.make_non_increasing(compressed_noisy_surv)

        noisy_probs = h.s2y(compressed_noisy_surv)
        surr_dataset = h.surrogates(bins, noisy_probs, ntot)

        return surr_dataset, noisy_probs, compressed_noisy_surv, bins

def _dp_probs(df, bin_length, epsilon):
    """
    minimal DP-Prob only for centralized setting with no censored data
    """
    max_time = math.floor(df["duration"].max())
    df_dig, bins = pp.digitize_time(df, bin_length, max_time)
    survival_values = h.df2s(df_dig, bins)
    ntot, n1, n0 = h.event_number_counter(df)
    probs = h.s2y(survival_values)
    sensitivity = 2/ntot
    noise = np.random.laplace(0, sensitivity/epsilon, size=probs.size)
    noisy_probs = np.clip(probs + noise, 0, None) #clip to [0,1]
    noisy_probs = noisy_probs/noisy_probs.sum() #normalize to sum up tp 1
    surrogate_df = h.surrogates(bins,
                                    noisy_probs,
                                    total_num=len(df))
    return surrogate_df

class CentralizedDPy:
    """
        args:
            df: pandas dataframe containing "duration" and "event" for each data
                point
            timestep (bins_length for preprocessing):int
                the length of time bins in real time, e.g. months
            max_time: int
                maximum time scale to be considered for the dataset
            epsilon: float
                privacy budget for DP with Laplace noise
            num: int
                if negative, sensitvity is calculated based on the df that is 
                passed as an argument. Otherwise in a non-uniform collaborative
                setting, it is calculated based on the size of the smallest 
                dataset i.e. num. 
    """
    def __init__(self, df, bin_length, max_time, epsilon):
        self.df = df
        self.bin_length = bin_length
        self.max_time = max_time
        self.epsilon = epsilon

    def DP_probs(self):
        """
        Applies local DP on y(t)

        returns:
            surrogate_df: pandas dataframe containing a surrogate dataset which
                has "event" and "duration" values for each data point in each row
        """

        df_dig, bins = pp.digitize_time(self.df, self.bin_length, self.max_time)
        survival_values = h.df2s(df_dig, bins)
        ntot, n1, n0 = h.event_number_counter(self.df)
        num_bins = np.shape(bins)[0]
        probs = h.s2y(survival_values)

        sensitivity = 2/n1
        noise = np.random.laplace(0, sensitivity/self.epsilon, size=probs.size)
        noisy_probs = np.clip(probs + noise, 0, None) #clip to [0,1]
        noisy_probs = noisy_probs/noisy_probs.sum() #normalize to sum up tp 1

        surrogate_df = h.surrogates(bins,
                                    noisy_probs,
                                    total_num=len(self.df))
        noisy_surv = h.y2s(noisy_probs)

        return surrogate_df, noisy_probs, noisy_surv, bins


def DP_matrix(df, bins_length, max_time, epsilon):
    """
    runs the DP-Matrix scheme for central setting

    args:
        df: Pandas dataframe of "duration" and "event" for each data point
        epsilon: float, privacy budget of the dp mechanism

    returns:
        noisy_df: pandas dataframe of "duration" and "event" for each data
            point with consideration of DP mechanism having been applied
    """
    df_dig, bins = pp.digitize_time(df, bins_length, max_time)
    survival_table, num = h.data2event_counter(df_dig)
    #laplace noise with sensitvity of 2 created for events of 1 and 0
    noise0 = np.random.laplace(0, 2.0/epsilon, size=len(survival_table))
    noise1 = np.random.laplace(0, 2.0/epsilon, size=len(survival_table))
    # noise_num = np.random.laplace(0, 2.0/epsilon, size=1)
    noisy0 = np.rint(survival_table["event0"] + noise0).astype(int)
    noisy1 = np.rint(survival_table["event1"] + noise1).astype(int)
    # noisy_num = np.rint(num + noise_num).astype(int)

    dpsurvival_table = pd.DataFrame({"duration": survival_table["duration"],
        "event1": noisy1,
        "event0": noisy0})
    #clip negative values
    dpsurvival_table = dpsurvival_table.clip(lower=0)

    #add the number of at risk column to dataframe
    dpsurvival_table = h.risk_counter(dpsurvival_table, num)
    noisy_df = h.count2duration(dpsurvival_table)

    return noisy_df, num



