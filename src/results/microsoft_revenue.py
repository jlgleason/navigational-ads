from functools import partial
from typing import List

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import fire
from cycler import cycler
from tqdm import tqdm

from google_revenue import get_ctr
from bca_boot import bca_boot

np.random.seed(42)
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["axes.prop_cycle"] = cycler(
    "color",
    [
        "#3f90da",
        "#ffa90e",
        "#bd1f01",
        "#94a4a2",
        "#832db6",
        "#a96b59",
        "#e76300",
        "#b9ac70",
        "#717581",
        "#92dadd",
    ],
)
mpl.rcParams["axes.axisbelow"] = True

FIG_FULL_WIDTH = (12, 4)
FIG_HALF_WIDTH = (5, 3)
FIG_THIRD_WIDTH = (4, 2.4)


def pre_plot(figsize):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.tick_params(which="both", bottom=False, left=False, right=False)
    plt.tight_layout()
    plt.grid(linestyle=":")


def post_plot(fp_output):
    for spine in ("top", "right", "bottom", "left"):
        plt.gca().spines[spine].set_visible(False)
    plt.savefig(fp_output, bbox_inches="tight")
    plt.close()


def get_qry_type(row):
    if row.nav:
        return "Navigational"
    elif row.brand:
        return "Brand"
    else:
        return "Non-Brand"


def get_device_ratio(row, col):
    if row[f"{col}_desktop"] == 0:
        return 0
    else:
        return row[f"{col}_mobile"] / row[f"{col}_desktop"]


def compute_phone_ratios(merged):
    """compute phone/desktop click/impression ratios to extrapolate to Microsoft revenue on phone"""

    merged["impression_ratio"] = merged.apply(
        partial(get_device_ratio, col="Impressions"), axis=1
    )
    return merged


def get_stats(topic, wt_col="weight", ratio_col="impression"):
    """computes user-weighted statistics for searches with topic==t"""

    text_ad = sum(topic.ad * topic[wt_col])
    shopping_ad = sum(topic.shopping_ads * topic[wt_col])
    impressions = text_ad + shopping_ad

    # CTR
    mask = topic.total_ads >= 0
    ctr = get_ctr(topic, mask, wt_col, "m_ad_clicks")

    # Revenue
    desktop_rev = (topic.m_ad_clicks * topic[wt_col] * topic.AverageCPC_desktop).sum()
    phone_rev = (
        topic.m_ad_clicks
        * topic[wt_col]
        * topic.AverageCPC_mobile
        * topic[f"{ratio_col}_ratio"]
    ).sum()

    return pd.Series(
        {
            "n_searches": topic[wt_col].sum(),
            "n_ad_clicks": sum(topic.m_ad_clicks * topic[wt_col]),
            "text_ad_impression_rate": text_ad / sum(topic[wt_col]),
            "shopping_ad_impression_rate": shopping_ad / sum(topic[wt_col]),
            "ad_impression_rate": impressions / sum(topic[wt_col]),
            "ad_ctr": ctr,
            "desktop_revenue": desktop_rev,
            "phone_revenue": phone_rev,
        }
    )


def compute_revenue_shares(data):

    data["fraction_searches"] = data.n_searches / data.n_searches.sum()
    data["desktop_revenue_share"] = data.desktop_revenue / data.desktop_revenue.sum()
    data["phone_revenue_share"] = data.phone_revenue / data.phone_revenue.sum()
    data["revenue_share"] = (data.desktop_revenue + data.phone_revenue) / (
        data.desktop_revenue + data.phone_revenue
    ).sum()
    return data


def bootstrap(data, n_boot, ratio_col):
    """bootstrap at user-level and compute stats"""

    users = data.user_id.drop_duplicates()
    results = []
    for _ in tqdm(range(n_boot)):

        # bootstrap sample over users
        bs_users = users.sample(frac=1, replace=True).value_counts()
        bs_users.name = "bs_weight"
        bs_sample = data.merge(bs_users, left_on="user_id", right_index=True)
        bs_sample["weight"] *= bs_sample["bs_weight"]

        # compute stats
        bs_results = bs_sample.groupby("topic").apply(
            partial(get_stats, ratio_col=ratio_col)
        )
        bs_results = compute_revenue_shares(bs_results)
        results.append(bs_results)

    results = pd.concat(results).reset_index()
    return results


def jackknife(data):
    """jackknife resample over users and compute stats"""
    results = []
    for user in tqdm(data.user_id.unique()):
        jk_sample = data[data.user_id != user]
        jk_results = jk_sample.groupby("topic").apply(get_stats)
        jk_results = compute_revenue_shares(jk_results)
        results.append(jk_results)

    results = pd.concat(results).reset_index()
    return results


def run(
    domains: List[str] = ["bing", "ddg"],
    n_boot: int = None,
    ratio_col: str = "impression",
):

    for domain in domains:

        sengine = pd.read_csv(f"data/{domain}.csv")
        sengine = compute_phone_ratios(sengine)

        # integer type for ad variables
        sengine[["ad", "shopping_ads"]] = sengine[["ad", "shopping_ads"]].fillna(0)
        sengine["total_ads"] = sengine.ad + sengine.shopping_ads
        cols = ["ad", "shopping_ads", "total_ads", "m_ad_clicks"]
        sengine[cols] = sengine[cols].astype(int)

        # original sample stats
        orig_stats = sengine.groupby("topic").apply(
            partial(get_stats, ratio_col=ratio_col)
        )
        orig_stats = compute_revenue_shares(orig_stats)

        # bootstrap, jackknife for BCa intervals
        if n_boot is not None:

            orig_stats.to_csv(f"data/{domain}_orig_stats.csv")

            bs_results = bootstrap(sengine, n_boot, ratio_col)
            bs_results.to_csv(f"data/{domain}_bs_results.csv", index=False)

            jk_results = jackknife(sengine)
            jk_results.to_csv(f"data/{domain}_jk_results.csv", index=False)


def plot(
    data,
    stats,
    names,
    fp_out,
    xlabel="",
    ylabel="",
    horizontal=True,
    ymax=None,
    xmax=None,
    adjust_font=False,
    qry_type="nav",
):

    data = data.melt(
        id_vars="sengine",
        value_vars=stats,
    )
    data = data.replace({stat: name for stat, name in zip(stats, names)})

    if horizontal:
        x = "value"
        y = "variable"
    else:
        x = "variable"
        y = "value"

    pre_plot(FIG_HALF_WIDTH)
    g = sns.barplot(
        data=data,
        x=x,
        y=y,
        hue="sengine",
        estimator=lambda x: np.median(x),
    )
    if adjust_font:
        g.tick_params(axis="x", labelsize=9)
        g.set_xlabel(xlabel=xlabel, fontsize=11)
        g.set_ylabel(ylabel=ylabel, fontsize=11)
    else:
        g.set(xlabel=xlabel, ylabel=ylabel)

    if qry_type == "Navigational":
        g.get_legend().remove()
    elif qry_type == "Brand":
        g.get_legend().set_title("")

    if ymax:
        g.set_ylim(0, ymax)
    if xmax:
        g.set_xlim(0, xmax)

    post_plot(fp_out)


def make_plots(
    fp_revenue: str = "figures/revenue_comparison",
    fp_ads: str = "figures/ad_comparison",
):

    # load data
    g = bca_boot(f"data/google", alpha=0.025, top_n_cats=10)
    g["sengine"] = "Google"
    b = bca_boot(f"data/bing", alpha=0.025, top_n_cats=3)
    b["sengine"] = "Bing"
    ddg = bca_boot(f"data/ddg", alpha=0.025, top_n_cats=3)
    ddg["sengine"] = "DuckDuckGo"
    data = pd.concat([g[b.columns], b, ddg])
    for qry_type in ["Navigational", "Brand"]:

        # revenue comparison
        plot(
            data[data.topic == qry_type],
            stats=[f"desktop_revenue_share", f"phone_revenue_share", f"revenue_share"],
            names=["Desktop", "Mobile", "Combined"],
            fp_out=f"{fp_revenue}_{qry_type.lower()}.pdf",
            xlabel="Modality",
            ylabel="Revenue Share",
            horizontal=False,
            adjust_font=True,
            qry_type=qry_type,
            ymax=0.4,
        )


if __name__ == "__main__":
    fire.Fire()
