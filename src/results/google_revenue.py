import multiprocessing
import os
from functools import partial
import string

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import fire
from cycler import cycler
from tqdm import tqdm

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

CATS = [
    "Abortion",
    "Advertising",
    "Advocacy Organizations",
    "Alcohol",
    "Alternative Beliefs",
    "Armed Forces",
    "Arts and Culture",
    "Auction",
    "Brokerage and Trading",
    "Business",
    "Charitable Organizations",
    "Child Education",
    "Child Sexual Abuse",
    "Content Servers",
    "Crypto Mining",
    "Dating",
    "Digital Postcards",
    "Discrimination",
    "Domain Parking",
    "Drug Abuse",
    "Dynamic Content",
    "Dynamic DNS",
    "Education",
    "Entertainment",
    "Explicit Violence",
    "Extremist Groups",
    "File Sharing and Storage",
    "Finance and Banking",
    "Folklore",
    "Freeware and Software Downloads",
    "Gambling",
    "Games",
    "General Organizations",
    "Global Religion",
    "Government and Legal Organizations",
    "Hacking",
    "Health and Wellness",
    "Homosexuality",
    "Illegal or Unethical",
    "Information Technology",
    "Information and Computer Security",
    "Instant Messaging",
    "Internet Radio and TV",
    "Internet Telephony",
    "Job Search",
    "Lingerie and Swimsuit",
    "Malicious Websites",
    "Marijuana",
    "Meaningless Content",
    "Medicine",
    "Newly Observed Domain",
    "Newly Registered Domain",
    "News and Media",
    "Newsgroups and Message Boards",
    "Not Rated",
    "Nudity and Risque",
    "Online Meeting",
    "Other Adult Materials",
    "Peer-to-peer File Sharing",
    "Personal Privacy",
    "Personal Vehicles",
    "Personal Websites and Blogs",
    "Phishing",
    "Plagiarism",
    "Political Organizations",
    "Pornography",
    "Proxy Avoidance",
    "Real Estate",
    "Reference",
    "Remote Access",
    "Restaurant and Dining",
    "Search Engines and Portals",
    "Secure Websites",
    "Sex Education",
    "Shopping",
    "Social Networking",
    "Society and Lifestyles",
    "Spam URLs",
    "Sports",
    "Sports Hunting and War Games",
    "Streaming Media and Download",
    "Terrorism",
    "Tobacco",
    "Travel",
    "URL Shortening",
    "Weapons (Sales)",
    "Web Analytics",
    "Web Chat",
    "Web Hosting",
    "Web-based Applications",
    "Web-based Email",
]

SHORT_ABBREV = {
    "Reference": "Reference",
    "News and Media": "News & Media",
    "Business": "Business",
    "Shopping": "Shopping",
    "Navigational": "Navigational",
    "Brand": "Brand",
    "Information Technology": "Info. Tech.",
    "Education": "Education",
    "Streaming Media and Download": "Streaming",
    "Entertainment": "Entertainment",
    "Health and Wellness": "Health",
    "Government and Legal Organizations": "Gov & Legal",
    "Social Networking": "Social Networking",
    "Restaurant and Dining": "Restaurants",
    "Sports": "Sports",
    "Finance and Banking": "Finance",
    "Travel": "Travel",
    "Games": "Games",
    "Society and Lifestyles": "Lifestyles",
    "Newsgroups and Message Boards": "Message Boards",
    "Real Estate": "Real Estate",
    "Personal Websites and Blogs": "Blogs",
}

LONG_ABBREV = {
    "Streaming Media and Download": "Streaming Media/Download",
    "Government and Legal Organizations": "Gov/Legal Organizations",
}


def pre_plot(ax):
    ax.tick_params(which="both", bottom=False, left=False, right=False)
    plt.tight_layout()
    plt.grid(linestyle=":")


def post_plot():
    for spine in ("top", "right", "bottom", "left"):
        plt.gca().spines[spine].set_visible(False)


def get_topic(row):
    """samples topic given P(c1, ..., c92)"""
    if row.nav:
        return "Navigational"
    elif row.brand:
        return "Brand"
    else:
        return np.random.choice(CATS, p=row[CATS])


def get_ctr(topic, mask, wt_col, click_col):
    """compute weighted CTR on subset defined by mask"""
    if sum(topic.loc[mask, "total_ads"]) == 0:
        ctr = 0
    else:
        wt_clicks = sum(topic.loc[mask, click_col] * topic.loc[mask, wt_col])
        wt_impressions = sum(topic.loc[mask, "total_ads"] * topic.loc[mask, wt_col])
        ctr = wt_clicks / wt_impressions

    return ctr


def get_stats(topic, wt_col="weight"):
    """computes user-weighted statistics for searches with specific topic"""

    text_ad = sum(topic.ad * topic[wt_col])
    shopping_ad = sum(topic.shopping_ads * topic[wt_col])
    impressions = text_ad + shopping_ad

    # CTR
    mask = topic.total_ads >= 0
    ctr = get_ctr(topic, mask, wt_col, "g_ad_clicks")

    return pd.Series(
        {
            f"n_searches": topic[wt_col].sum(),
            f"n_ad_clicks": sum(topic.g_ad_clicks * topic[wt_col]),
            f"ad_impression_rate": impressions / sum(topic[wt_col]),
            f"text_ad_impression_rate": text_ad / sum(topic[wt_col]),
            f"shopping_ad_impression_rate": shopping_ad / sum(topic[wt_col]),
            f"ad_ctr": ctr,
            f"desktop_revenue": topic["desktop_rev"].sum(),
            f"phone_revenue": topic["phone_rev"].sum(),
        }
    )


def get_device_ratio(row):
    if row.DESKTOP == 0:
        return float("nan")
    else:
        return row.MOBILE / row.DESKTOP


def cpc_dist(merged, top_topics, fp_cpc):
    """plot CPC distributions across categories"""

    mask = (
        (merged.total_ads > 0)
        & (merged.average_cpc > 0)
        & (merged.topic.isin(top_topics))
    )
    cpc_data = merged[mask].copy()
    cpc_data["topic_order"] = cpc_data.topic.apply(
        lambda t: top_topics.tolist().index(t)
    )
    cpc_data = cpc_data.sort_values(by="topic_order")
    mean_cpc = (
        cpc_data.groupby("topic")["average_cpc"]
        .apply(lambda x: x.mean())
        .sort_values(ascending=False)
    )

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    h = 2 * 4 / len(top_topics)
    g = sns.FacetGrid(
        cpc_data,
        row="topic",
        hue="topic",
        aspect=4 / h,
        height=h,
        row_order=mean_cpc.index,
    )
    g.map(
        sns.kdeplot,
        "average_cpc",
        weights=cpc_data.weight,
        fill=True,
        lw=1,
        alpha=1,
        log_scale=True,
        bw_adjust=0.5,
    )
    g.map(
        sns.kdeplot,
        "average_cpc",
        weights=cpc_data.weight,
        color="w",
        lw=1.5,
        bw_adjust=0.5,
    )

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(
            -0.35,
            0.2,
            SHORT_ABBREV[label],
            fontsize="x-small",
            color=color,
            ha="left",
            va="center",
            transform=ax.transAxes,
        )

    g.map(label, "topic")

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-0.5)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(
        yticks=[],
        ylabel="",
        xlabel="Average CPC (USD)",
        xlim=(10**-1, 10**2),
    )
    g.despine(bottom=True, left=True)

    plt.savefig(fp_cpc, bbox_inches="tight")
    plt.close()


def eval_selection_bias(merged, top_topics, fp_selection):
    """evaluate selection bias from bid price matching"""

    g_api_mask = merged.device_sum > 0

    # matched SERPs
    matched_stats = (
        merged[g_api_mask]
        .groupby("topic")
        .apply(partial(get_stats, wt_col="equal_weight"))
        .reset_index()
    )
    matched_stats["fraction_searches"] = matched_stats["n_searches"] / sum(
        matched_stats["n_searches"]
    )
    matched_stats["fraction_ad_clicks"] = matched_stats["n_ad_clicks"] / sum(
        matched_stats["n_ad_clicks"]
    )
    matched_stats["subset"] = "searches with API data"

    # unmatched SERPs
    unmatched_stats = (
        merged[~g_api_mask]
        .groupby("topic")
        .apply(partial(get_stats, wt_col="equal_weight"))
        .reset_index()
    )
    unmatched_stats["fraction_searches"] = unmatched_stats["n_searches"] / sum(
        unmatched_stats["n_searches"]
    )
    unmatched_stats["fraction_ad_clicks"] = unmatched_stats["n_ad_clicks"] / sum(
        unmatched_stats["n_ad_clicks"]
    )
    unmatched_stats["subset"] = "searches without API data"

    # plot
    xvals = ["fraction_ad_clicks"]
    labels = ["Fraction of Ad Clicks"]
    data = pd.concat([matched_stats, unmatched_stats])
    data = data[data.topic.isin(top_topics)]
    data = data.replace(SHORT_ABBREV)
    top_topics = top_topics.replace(SHORT_ABBREV)

    fig = plt.figure(figsize=(5, 4))
    for i, (xval, label) in enumerate(zip(xvals, labels)):
        ax = fig.add_subplot(1, len(xvals), i + 1)
        pre_plot(ax)
        g = sns.barplot(data=data, x=xval, y="topic", order=top_topics, hue="subset")
        post_plot()
        if i > 0:
            g.set_yticklabels([])
            g.set(xlabel=label, ylabel="")
            g.get_legend().remove()
        else:
            g.set(xlabel=label, ylabel="Category")
            g.get_legend().set_title("")
            plt.savefig(fp_selection, bbox_inches="tight")
    plt.close()


def get_qry_type(row):
    if row.nav:
        return "Navigational"
    elif row.brand:
        return "Brand"
    else:
        return "Non-Brand"


def plot_phone_ratio(merged):
    """plot mobile/desktop search volume ratio for nav/brand/non-brand"""

    fig = plt.figure(figsize=(5, 2.5))
    ax = fig.add_subplot(111)
    pre_plot(ax)

    g = sns.ecdfplot(
        data=merged[merged.device_ratio > 0],
        x="device_ratio",
        hue="qry_type",
        log_scale=True,
        hue_order=["Navigational", "Brand", "Non-Brand"],
    )
    g.set(xlim=(10**-1, 10**1))
    g.set(xlabel=f"Mobile / Desktop Search Volume Ratio")
    g.get_legend().set_title("")
    post_plot()

    plt.savefig("figures/search_ratio.pdf", bbox_inches="tight")
    plt.close()


def bootstrap(data, n_boot):
    """bootstrap at user-level and compute stats"""

    users = data.user_id.drop_duplicates()
    results = []
    for _ in tqdm(range(n_boot)):

        # bootstrap sample over users
        bs_users = users.sample(frac=1, replace=True).value_counts()
        bs_users.name = "bs_weight"
        bs_sample = data.merge(bs_users, left_on="user_id", right_index=True)
        bs_sample["weight"] *= bs_sample["bs_weight"]

        # sample topic for each SERP
        bs_sample = sample_topics_parallel(bs_sample, get_topic)
        bs_sample = compute_pointwise_revenue(bs_sample)

        # compute stats
        bs_results = bs_sample.groupby("topic").apply(get_stats)
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


def compute_pointwise_revenue(data, wt_col="weight"):
    data["desktop_rev"] = data.g_ad_clicks * data[wt_col] * data.average_cpc
    data["phone_rev"] = data.desktop_rev * data.device_ratio
    return data


def compute_revenue_shares(data):
    """compute revenue shares and fraction of total searches"""

    # fraction searches
    data["fraction_searches"] = data.n_searches / data.n_searches.sum()

    # revenue shares
    for prefix in ["desktop", "phone"]:
        data[f"{prefix}_revenue_share"] = (
            data[f"{prefix}_revenue"] / data[f"{prefix}_revenue"].sum()
        )

    data[f"revenue_share"] = (data.desktop_revenue + data.phone_revenue) / (
        data.desktop_revenue + data.phone_revenue
    ).sum()

    return data


def plot(data, top_topics, fp_output, xvals, labels):
    """plots multiple horizontal barplots"""
    fig = plt.figure(figsize=(12, 5))
    for i, (xval, label) in enumerate(zip(xvals, labels)):
        ax = fig.add_subplot(1, len(xvals), i + 1)

        pre_plot(ax)
        g = sns.barplot(
            data=data,
            x=xval,
            y="topic",
            order=top_topics,
            estimator=lambda x: np.median(x),
        )

        b, t = ax.get_ylim()
        l, r = ax.get_xlim()
        ax.text((l + r) / 2, b + 3.5, f"({string.ascii_lowercase[i]})")

        if i > 0:
            g.set_yticklabels([])
            g.set(xlabel=label, ylabel="")
        else:
            g.set(xlabel=label, ylabel="Category")

        post_plot()

    plt.savefig(fp_output, bbox_inches="tight")
    plt.close()


def get_df(results, cols):
    cols += ["topic"]
    data = results.loc[results.topic.isin(["Navigational", "Brand"]), cols]
    data = data.melt(id_vars="topic")
    data["ad_click"] = data.variable.replace({cols[0]: "Focal", cols[1]: "Competitor"})
    return data


def sample_topics_subset(data_subset, func):
    """samples topics for a subset of rows"""
    data_subset["topic"] = data_subset.apply(func, axis=1)
    return data_subset


def sample_topics_parallel(data, func):
    """samples topics in parallel"""
    n_proc = os.cpu_count() - 2
    data_split = np.array_split(data, n_proc)
    with multiprocessing.Pool(n_proc) as pool:
        results = pool.map(partial(sample_topics_subset, func=func), data_split)
    return pd.concat(results)


def run(
    fp_data: str = "data/google.csv",
    fp_cpc: str = "figures/cpc_dist.pdf",
    fp_selection: str = "figures/selection.pdf",
    n_boot: int = None,
    top_n_categories: int = 20,
):

    os.makedirs("figures", exist_ok=True)

    merged = pd.read_csv(fp_data)
    merged["equal_weight"] = 1
    merged["device_ratio"] = merged.apply(get_device_ratio, axis=1)
    plot_phone_ratio(merged)

    # original sample stats
    merged = sample_topics_parallel(merged, get_topic)
    merged = compute_pointwise_revenue(merged)

    orig_stats = merged.groupby("topic").apply(get_stats)
    orig_stats = compute_revenue_shares(orig_stats)

    # eval selection bias, plot bid price dist.
    top_fracs = orig_stats.sort_values(by="fraction_searches", ascending=False)
    top_fracs = top_fracs.head(top_n_categories).reset_index()
    eval_selection_bias(merged, top_fracs.topic, fp_selection)
    cpc_dist(merged, top_fracs.topic, fp_cpc)

    # bootstrap, jackknife for BCa intervals
    if n_boot is not None:

        orig_stats.to_csv(f"data/google_orig_stats.csv")

        bs_results = bootstrap(merged, n_boot)
        bs_results.to_csv(f"data/google_bs_results.csv", index=False)

        jk_results = jackknife(merged)
        jk_results.to_csv(f"data/google_jk_results.csv", index=False)


def make_plots(
    fp_input: str = "data/google",
    fp_plots: str = "figures/google",
    top_n_cats: int = 20,
):

    # take top_n categories
    data = bca_boot(fp_input, alpha=0.025, top_n_cats=top_n_cats)
    data = data.replace(LONG_ABBREV)
    top_topics = (
        data.drop_duplicates(subset="topic", keep="first")
        .sort_values(by="n_searches", ascending=False)
        .topic.head(top_n_cats)
    )

    # plots
    plot(
        data,
        top_topics,
        f"{fp_plots}_desktop_ads.pdf",
        xvals=[
            "fraction_searches",
            "text_ad_impression_rate",
            "shopping_ad_impression_rate",
            "ad_ctr",
        ],
        labels=[
            "Fraction of Searches",
            "Text Ad Impression Rate",
            "Shopping Ad Impression Rate",
            "Ad Click-through Rate",
        ],
    )
    plot(
        data,
        top_topics,
        f"{fp_plots}_revenue.pdf",
        xvals=["desktop_revenue_share", "phone_revenue_share", "revenue_share"],
        labels=[
            "Desktop Revenue Share",
            "Mobile Revenue Share",
            "Combined Revenue Share",
        ],
    )


if __name__ == "__main__":
    fire.Fire()
