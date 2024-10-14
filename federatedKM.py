import numpy as np
import pandas as pd
import math

import centralizedDP as cdp
import helpers as h

###The federated part of the code, in the current format

def data_split_uniform(df, num_clients):
    """
    splits the data evenly among num_clients clients

    args:
        df: pandas dataframe to be shuffled and split
        num_clients: int
                the number of clients
    returns:
        data_split: df with num_clients indices
    """
    df = df.sample(frac=1, random_state=24)  # shuffle
    max_idx = (len(df)//num_clients)*num_clients
    n = int(max_idx/num_clients)
    df_split = [df[i:i+n] for i in range(0, max_idx, n)]
    return df_split

def data_split_nonuniform(df, num_clients, frac_one):
    """
        splits the data non-uniformly, one site has more/less data

        args:
        df: pandas dataframe to be split
        num_clients: int
                total number of clients in the system
        frac_one: float
                fraction of data that one client receives. The rest of the
                data will be split evenly among num_clients-1
        returns:
        data_split: dataframe where first index is the client with diff data
        """
    df = df.sample(frac=1, random_state=24)  # shuffle
    ind1 = math.floor(frac_one * len(df))
    ind2 = math.floor(((1 - frac_one) / (num_clients-1)) * len(df))
    end = ind1 + (ind2 * (num_clients-1))
    indices = np.linspace(ind1, end, num_clients, dtype=int)
    id = np.insert(indices, 0, 0)
    data_split = [df[id[i]:id[i+1]] for i in range(0, num_clients, 1)]
    # data_split = np.split(df[:end], indices[:-1])
    return data_split

def fed_DPMatrix_pooledData(df, num_client, bin_length, max_time, epsilon, frac_one=-1):
    """
    runs DP-surv with the pooled data view. 

    args:
        df: the complete dataset to be split among clients
        num_clients: int, number of clients
        bin_length: int, the length of binning of dataset, will be the same 
            across all clients for consistency
        max_time: int, maximum time-scale for the dataset, will be the same 
            across all clients
        frac: float, fraction of first coefficients of DCT to take out of all
        epsilon: float, privacy budget considered for each client
        frac_one: float, 
            if -1 a uniform data splitting is performed, otherwise, the minority
            site will have frac_one of the data, e.g. 0.50 or 0.05
    returns:
        mean_data: dataframe, the collaborative and private surrogate dataset 
        formed over this path. 
    """
    if frac_one < 0:
        data_split = data_split_uniform(df, num_client)
    else:
        data_split = data_split_nonuniform(df, num_client, frac_one)

    #minimum size of the uncencored dataset among all clients
    # n = np.min([data_split[i]["event"].sum() for i in np.arange(num_client)])

    mean_data = pd.DataFrame([])
    for i in np.arange(num_client):
        surr, _  = cdp.DP_matrix(data_split[i], bin_length, max_time, epsilon)
        mean_data = pd.concat([mean_data, surr])
    return mean_data

def fed_DPS_pooledData(df, num_client, bin_length, max_time,
                       frac, epsilon, frac_one=-1):
    """
    runs DP-surv with the pooled data view. 

    args:
        df: the complete dataset to be split among clients
        num_clients: int, number of clients
        bin_length: int, the length of binning of dataset, will be the same 
            across all clients for consistency
        max_time: int, maximum time-scale for the dataset, will be the same 
            across all clients
        frac: float, fraction of first coefficients of DCT to take out of all
        epsilon: float, privacy budget considered for each client
        frac_one: float, 
            if -1 a uniform data splitting is performed, otherwise, the minority
            site will have frac_one of the data, e.g. 0.50 or 0.05
    returns:
        mean_data: dataframe, the collaborative and private surrogate dataset 
        formed over this path. 
    """
    if frac_one < 0:
        data_split = data_split_uniform(df, num_client)
    else:
        data_split = data_split_nonuniform(df, num_client, frac_one)

    #minimum size of the uncencored dataset among all clients
    # n = np.min([data_split[i]["event"].sum() for i in np.arange(num_client)])

    mean_data = pd.DataFrame([])
    for i in np.arange(num_client):
        c = cdp.CentralizedDPS(data_split[i], bin_length, max_time, frac, epsilon)
        surr, _, _, _ = c.DP_Surv()
        mean_data = pd.concat([mean_data, surr])
    return mean_data


def fed_DPS_avgS(df, num_client, bin_length, max_time,
                 frac, epsilon, frac_one=-1):
    """
    runs DP-surv with the averaged private S(t) view. 

    args:
        df: the complete dataset to be split among clients
        num_clients: int, number of clients
        bin_length: int, the length of binning of dataset, will be the same 
            across all clients for consistency
        max_time: int, maximum time-scale for the dataset, will be the same 
            across all clients
        frac: float, fraction of first coefficients of DCT to take out of all
        epsilon: float, privacy budget considered for each client
        frac_one: float, 
            if -1 a uniform data splitting is performed, otherwise, the minority
            site will have frac_one of the data, e.g. 0.50 or 0.05
    returns:
        mean_data: dataframe, the collaborative and private surrogate dataset 
        formed over this path. 
        mean_surv: float, the array (size bins) of averaged private survival 
        values obtained through collaboration 
        mean_y: float, array of averaged private probability values obtained in
        the collaborative setting
    """
    if frac_one < 0:
        data_split = data_split_uniform(df, num_client)
    else:
        data_split = data_split_nonuniform(df, num_client, frac_one)
    
    #minimum size of the uncencored dataset among all clients
    # n = np.min([data_split[i]["event"].sum() for i in np.arange(num_client)])

    c = cdp.CentralizedDPS(data_split[0], bin_length, max_time,
                               frac, epsilon)
    _, _, surv, bins = c.DP_Surv()
    all_surv = np.empty([num_client, len(surv)])
    all_surv[0] = surv

    for i in np.arange(1, num_client):
        c = cdp.CentralizedDPS(data_split[i], bin_length, max_time,
                               frac, epsilon)
        _, _, surv, _ = c.DP_Surv()
        all_surv[i] = surv
    w = [len(data_split[i]) for i in range(0, num_client, 1)]

    mean_surv = np.average(all_surv, weights=w, axis=0)
    mean_y = h.s2y(mean_surv)
    mean_data = h.surrogates(bins, mean_y, len(df))
    return mean_data, mean_surv, mean_y
        
def fed_DPS_avgy(df, num_client, bin_length, max_time,
                 frac, epsilon, frac_one=-1):
    """
    runs DP-surv with the averaged private y(t) view. 

    args:
        df: the complete dataset to be split among clients
        num_clients: int, number of clients
        bin_length: int, the length of binning of dataset, will be the same 
            across all clients for consistency
        max_time: int, maximum time-scale for the dataset, will be the same 
            across all clients
        frac: float, fraction of first coefficients of DCT to take out of all
        epsilon: float, privacy budget considered for each client
        frac_one: float, 
            if -1 a uniform data splitting is performed, otherwise, the minority
            site will have frac_one of the data, e.g. 0.50 or 0.05
    returns:
        mean_data: dataframe, the collaborative and private surrogate dataset 
        formed over this path. 
        mean_y: float, array of averaged private probability values obtained in
        the collaborative setting
    """
    if frac_one < 0:
        data_split = data_split_uniform(df, num_client)
    else:
        data_split = data_split_nonuniform(df, num_client, frac_one)

    #minimum size of the uncencored dataset among all clients
    # n = np.min([data_split[i]["event"].sum() for i in np.arange(num_client)])

    c = cdp.CentralizedDPS(data_split[0], bin_length, max_time,
                               frac, epsilon)
    surra, prob, _, bins = c.DP_Surv()
    all_prob = np.empty([num_client, len(prob)])
    all_prob[0] = prob

    for i in np.arange(1, num_client):
        c = cdp.CentralizedDPS(data_split[i], bin_length, max_time,
                               frac, epsilon)
        _, prob, _, _ = c.DP_Surv()
        all_prob[i] = prob

    w = [len(data_split[i]) for i in range(0, num_client, 1)] #weight vector 

    mean_y = np.average(all_prob, weights=w, axis=0)
    mean_data = h.surrogates(bins, mean_y, len(df))
    return mean_data, mean_y

def fed_DPy_pooledData(df, num_client, bin_length, max_time,
                       epsilon, frac_one=-1):
    """
    runs DP-Prob with the pooled data view. 

    args:
        df: the complete dataset to be split among clients
        num_clients: int, number of clients
        bin_length: int, the length of binning of dataset, will be the same 
            across all clients for consistency
        max_time: int, maximum time-scale for the dataset, will be the same 
            across all clients
        epsilon: float, privacy budget considered for each client
        frac_one: float, 
            if -1 a uniform data splitting is performed, otherwise, the minority
            site will have frac_one of the data, e.g. 0.50 or 0.05
    returns:
        mean_data: dataframe, the collaborative and private surrogate dataset 
        formed over this path. 
    """
    if frac_one < 0:
        data_split = data_split_uniform(df, num_client)
    else:
        data_split = data_split_nonuniform(df, num_client, frac_one)

    mean_data = pd.DataFrame([])
    for i in np.arange(num_client):
        c = cdp.CentralizedDPy(data_split[i], bin_length, max_time,
                               epsilon)
        surr, _, _, _ = c.DP_probs()
        mean_data = pd.concat([mean_data, surr])
    return mean_data


def fed_DPy_avgS(df, num_client, bin_length, max_time,
                 epsilon, frac_one=-1):
    """
    runs DP-Prob with the averaged private S(t) view. 

    args:
        df: the complete dataset to be split among clients
        num_clients: int, number of clients
        bin_length: int, the length of binning of dataset, will be the same 
            across all clients for consistency
        max_time: int, maximum time-scale for the dataset, will be the same 
            across all clients
        epsilon: float, privacy budget considered for each client
        frac_one: float, 
            if -1 a uniform data splitting is performed, otherwise, the minority
            site will have frac_one of the data, e.g. 0.50 or 0.05
    returns:
        mean_data: dataframe, the collaborative and private surrogate dataset 
        formed over this path. 
        mean_surv: float, the array (size bins) of averaged private survival 
        values obtained through collaboration 
        mean_y: float, array of averaged private probability values obtained in
        the collaborative setting
    """
    if frac_one < 0:
        data_split = data_split_uniform(df, num_client)
    else:
        data_split = data_split_nonuniform(df, num_client, frac_one)
    
    #minimum size of the uncencored dataset among all clients

    c = cdp.CentralizedDPy(data_split[0], bin_length, max_time,
                           epsilon)
    surra, _, surv, bins = c.DP_probs()
    all_surv = np.empty([num_client, len(surv)])
    all_surv[0] = surv

    for i in np.arange(1, num_client):
        c = cdp.CentralizedDPy(data_split[i], bin_length, max_time,
                               epsilon)
        _, _, surv, _ = c.DP_probs()
        all_surv[i] = surv

    w = [len(data_split[i]) for i in range(0, num_client, 1)]
    mean_surv = np.average(all_surv, weights=w, axis=0)
    mean_y = h.s2y(mean_surv)
    mean_data = h.surrogates(bins, mean_y, len(df))
    return mean_data, mean_surv, mean_y
        

def fed_DPy_avgy(df, num_client, bin_length, max_time,
                 epsilon, frac_one=-1):
    """
    runs DP-Prob with the averaged private y(t) view. 

    args:
        df: the complete dataset to be split among clients
        num_clients: int, number of clients
        bin_length: int, the length of binning of dataset, will be the same 
            across all clients for consistency
        max_time: int, maximum time-scale for the dataset, will be the same 
            across all clients
        epsilon: float, privacy budget considered for each client
        frac_one: float, 
            if -1 a uniform data splitting is performed, otherwise, the minority
            site will have frac_one of the data, e.g. 0.50 or 0.05
    returns:
        mean_data: dataframe, the collaborative and private surrogate dataset 
        formed over this path. 
        mean_y: float, array of averaged private probability values obtained in
        the collaborative setting
    """
    if frac_one < 0:
        data_split = data_split_uniform(df, num_client)
    else:
        data_split = data_split_nonuniform(df, num_client, frac_one)

    c = cdp.CentralizedDPy(data_split[0], bin_length, max_time,
                               epsilon)
    surra, prob, _, bins = c.DP_probs()
    all_prob = np.empty([num_client, len(prob)])
    all_prob[0] = prob

    for i in np.arange(1, num_client):
        c = cdp.CentralizedDPy(data_split[i], bin_length, max_time,
                               epsilon)
        _, prob, _, _ = c.DP_probs()
        all_prob[i] = prob

    w = [len(data_split[i]) for i in range(0, num_client, 1)]
    mean_y = np.average(all_prob, weights=w, axis=0)
    mean_data = h.surrogates(bins, mean_y, len(df))
    return mean_data, mean_y
