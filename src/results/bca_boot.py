import math

from scipy.special import ndtri, ndtr
import numpy as np
import pandas as pd

# "bootstrapped" from scipy.stats.bootstrap implementation


def prep_subset(orig, bs_samples, jk_samples, topic):

    bs_subset = bs_samples[bs_samples.topic == topic].drop(columns="topic")
    nb = len(bs_subset)
    bs_subset = bs_subset.reset_index(drop=True)

    jk_subset = jk_samples[jk_samples.topic == topic].drop(columns="topic")
    nj = len(jk_subset)
    jk_subset = jk_subset.reset_index(drop=True)

    orig = pd.concat([orig[orig.topic == topic]] * nb).drop(columns="topic")
    orig = orig.reset_index(drop=True)

    return orig, bs_subset, jk_subset, nb, nj


def robust_pct(bs_subset, alpha_1, alpha_2, col):
    if math.isnan(alpha_1[col]):
        return (np.nan, np.nan)
    else:
        return np.percentile(bs_subset[col], (alpha_1[col] * 100, alpha_2[col] * 100))


def prep_output(bs_subset, topic, alpha_1, alpha_2):
    cols = bs_subset.columns
    pcts = [robust_pct(bs_subset, alpha_1, alpha_2, col) for col in cols]
    bca_lo = {col: lo for col, (lo, _) in zip(cols, pcts)} | {"topic": topic}
    bca_hi = {col: hi for col, (_, hi) in zip(cols, pcts)} | {"topic": topic}
    data = pd.DataFrame([bca_lo, bca_hi])
    return data


def percentile_of_score(boot, orig, nb):
    return ((boot < orig).sum() + (boot <= orig).sum()) / (2 * nb)


def bias_param(orig, bs_subset, nb):
    pct = percentile_of_score(bs_subset, orig, nb)
    z0_hat = ndtri(pct)
    return z0_hat


def accel_param(jk_subset, nj):

    U_i = (nj - 1) * (jk_subset.mean(numeric_only=True) - jk_subset)
    num = (U_i**3).sum() / nj**3
    denom = (U_i**2).sum() / nj**2
    a_hat = 1 / 6 * num / denom ** (3 / 2)
    return a_hat


def get_alphas(alpha, z0_hat, a_hat):
    z_alpha = ndtri(alpha)
    z_1alpha = -z_alpha
    num1 = z0_hat + z_alpha
    alpha_1 = ndtr(z0_hat + num1 / (1 - a_hat * num1))
    num2 = z0_hat + z_1alpha
    alpha_2 = ndtr(z0_hat + num2 / (1 - a_hat * num2))
    return alpha_1, alpha_2


def bca_boot(fp_data, alpha=0.025, top_n_cats=20):
    orig_stats = pd.read_csv(f"{fp_data}_orig_stats.csv")
    bs_samples = pd.read_csv(f"{fp_data}_bs_results.csv")
    jk_samples = pd.read_csv(f"{fp_data}_jk_results.csv")
    top_topics = orig_stats.sort_values(by="n_searches", ascending=False)
    bca_results = []
    for topic in top_topics.topic.iloc[:top_n_cats]:
        orig, bs_subset, jk_subset, nb, nj = prep_subset(
            orig_stats, bs_samples, jk_samples, topic
        )
        z0_hat = bias_param(orig, bs_subset, nb)
        a_hat = accel_param(jk_subset, nj)
        alpha_1, alpha_2 = get_alphas(alpha, z0_hat, a_hat)

        res = prep_output(bs_subset, topic, alpha_1, alpha_2)
        bca_results.append(res)

    results = pd.concat([orig_stats] + bca_results)
    return results
