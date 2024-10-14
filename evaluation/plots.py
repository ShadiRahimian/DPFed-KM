import numpy as np
import pandas as pd

import evaluation.evaluation as ev
import preprocessing as pp
import helpers as h

from lifelines import statistics as stat
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times


def plot_recons(df, bins_length, max_time, n, tarr):
    km = KaplanMeierFitter()
    kmreal = km.fit(df["duration"], df["event"], label=f"real, n={n}")
    mreal = round(kmreal.median_survival_time_)

    df_dig, bins = pp.digitize_time(df, bins_length, max_time)
    s = h.df2s(df_dig, bins)
    y = h.s2y(s)
    surr = h.surrogates(bins, y, n)
    p = stat.logrank_test(df["duration"], surr["duration"], df["event"], surr["event"])
    p = np.round(p.p_value, 2)
    kmf = KaplanMeierFitter()
    kmfr = kmf.fit(surr["duration"], surr["event"], label=f"b={bins_length}, n={n}")
    m = round(kmfr.median_survival_time_)
    percent_diff_median = np.round(abs(m - mreal) / mreal, 3)
    y_upp, y_low = ev.confidence_at_time(tarr, kmfr)
    return kmfr, p, m, percent_diff_median, y_upp, y_low


def plots_centeralized(df, surr, tarr):
    kmf = KaplanMeierFitter()
    kmreal = kmf.fit(df["duration"], df["event"], label=f"real")
    mreal = round(kmreal.median_survival_time_)

    kmfr = kmf.fit(surr["duration"], surr["event"])
    m = round(kmfr.median_survival_time_)
    # median_ci = median_survival_times(kmf.confidence_interval_)
    # median_ci = median_ci.reset_index(level=0)
    # median_ci = np.array(median_ci)
    cmd = np.round(abs(m - mreal) / mreal, 3)

    p = stat.logrank_test(df["duration"], surr["duration"], df["event"], surr["event"])
    p = np.round(p.p_value, 2)

    # y = np.array(kmfr.survival_function_at_times(tarr)) #value of KM at tarr
    # y = np.around(y, 2)
    # yu, yl = ev.confidence_at_time(tarr, kmfr)
    # yu = np.around(yu, 2)
    # yl = np.around(yl, 2)
    # yuerr = yu -y #upper error given to pyplot
    # ylerr = y - yl #lower error given to pyplot
    return p, m, cmd

def median(df):
    kmf = KaplanMeierFitter()
    kmreal = kmf.fit(df["duration"], df["event"])
    mreal = round(kmreal.median_survival_time_)
    return mreal

def fed_plots(bin_length, max_time, epsilon, frac, num_client, percent_one, title):   

    sdp, sap = fdk.fed_DPS_pooledData(df, num_client, bin_length, max_time, frac, epsilon, percent_one)
    sds, _, _, sas = fdk.fed_DPS_avgS(df, num_client, bin_length, max_time, frac, epsilon, percent_one)
    sdy, _, say = fdk.fed_DPS_avgy(df, num_client, bin_length, max_time, frac, epsilon, percent_one)

    ydp, yap = fdk.fed_DPy_pooledData(df, num_client, bin_length, max_time, epsilon, percent_one)
    yds, _, _, yas = fdk.fed_DPy_avgS(df, num_client, bin_length, max_time, epsilon, percent_one)
    ydy, _, yay = fdk.fed_DPy_avgy(df, num_client, bin_length, max_time, epsilon, percent_one)

    ax = plt.subplot(111)
    kmf = KaplanMeierFitter()
    a = kmf.fit(df["duration"], df["event"], label="Original")
    m = a.median_survival_time_
    a.plot(color="#1f77b4")

    c1 = kmf.fit(sap["duration"], sap["event"], label="Site 1")
    m0 = c1.median_survival_time_
    p0 = plots.pvalue(df, sap)
    c1.plot(ci_force_lines=False, ci_show=False, color="r", linestyle="-")

    asp = kmf.fit(sdp["duration"], sdp["event"], label="DP-S Pooled")
    m1 = asp.median_survival_time_
    p1 = plots.pvalue(df, sdp)
    asp.plot(ci_force_lines=False, ci_show=False, color="#ff7f0e", linestyle="-")
    ass = kmf.fit(sds["duration"], sds["event"], label="DP-S avg S")
    m2 = ass.median_survival_time_
    p2 = plots.pvalue(df, sds)
    ass.plot(ci_force_lines=False, ci_show=False, color="#ff7f0e", linestyle="--")
    asy = kmf.fit(sdy["duration"], sdy["event"], label="DP-S avg prob")
    m3 = asy.median_survival_time_
    p3 = plots.pvalue(df, sdy)
    asy.plot(ci_force_lines=False, ci_show=False, color="#ff7f0e", linestyle=":")

    ayp = kmf.fit(ydp["duration"], ydp["event"], label="DP-y Pooled")
    m4 = ayp.median_survival_time_
    p4 = plots.pvalue(df, ydp)
    ayp.plot(ci_force_lines=False, ci_show=False, color="g", linestyle="-")
    ays = kmf.fit(yds["duration"], yds["event"], label="DP-y avg S")
    m5 = ays.median_survival_time_
    p5 = plots.pvalue(df, yds)
    ays.plot(ci_force_lines=False, ci_show=False, color="g", linestyle="--")
    ayy = kmf.fit(ydy["duration"], ydy["event"], label="DP-y avg prob")
    m6 = ayy.median_survival_time_
    p6 = plots.pvalue(df, ydy)
    ayy.plot(ci_force_lines=False, ci_show=False, color="g", linestyle=":")

    ax.set_title(f"{name}, {title}, $\epsilon$={epsilon}")
    ax.set_xlabel("Time", fontsize="large")
    ax.set_ylabel("Survival Probability", fontsize="large")
    leg1 = ax.legend()
    leg2 = ax.legend([f"m={round(m)}, p=1", f"m={round(m0)}, p={p0}",
                    f"m={round(m1)}, p={p1}",
                    f"m={round(m2)}, p={p2}", f"m={round(m3)}, p={p3}",
                    f"m={round(m4)}, p={p4}", f"m={round(m5)}, p={p5}",
                    f"m={round(m6)}, p={p6}"], bbox_to_anchor=(1.4, 1.))
    ax.add_artist(leg1)
    ax.add_artist(leg2)
    plt.savefig(f'figs/{name}-{title}-fed-b{bin_length}-e{epsilon}.pdf', dpi=300, bbox_inches="tight")
    plt.show()