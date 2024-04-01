import os
import string

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels import stats
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler

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


def pre_plot():
    plt.gca().tick_params(which="both", bottom=False, left=False, right=False)
    plt.tight_layout()
    plt.grid(linestyle=":")


def post_plot():
    for spine in ("top", "right", "bottom", "left"):
        plt.gca().spines[spine].set_visible(False)


def simonov_baselines():
    simonov_comp = pd.DataFrame(
        [
            [0.0034313725490196234, 0.004901960784313736, 0, True],
            [0.009313725490196073, 0.014215686274509809, 1, True],
            [0.013235294117647067, 0.02745098039215685, 2, True],
            [0.016176470588235292, 0.0215686274509804, 3, True],
            [0.004901960784313736, 0.0058823529411764774, 0, False],
            [0.05588235294117645, 0.07156862745098037, 1, False],
            [0.09411764705882351, 0.1068627450980392, 2, False],
            [0.11813725490196077, 0.1333333333333333, 3, False],
            # [0.13970588235294115, 0.15931372549019604, 4, False],
        ],
        columns=["lo", "hi", "n_comps", "focal_ad_top"],
    )
    simonov_comp = simonov_comp.melt(
        id_vars=["n_comps", "focal_ad_top"], value_vars=["lo", "hi"]
    )

    simonov_comp_hq = pd.DataFrame(
        [
            [0.0023282031173414153, 0.00330849916674833, 0, True],
            [0.005759239290265644, 0.009190275463189873, 1, True],
            [0.007229683364376016, 0.013601607685521017, 2, True],
            [0.007719831389079501, 0.016542495833741788, 3, True],
            [0.003960396039603936, 0.003960396039603936, 0, False],
            [0.02772277227722772, 0.03415841584158413, 1, False],
            [0.04752475247524751, 0.05396039603960395, 2, False],
            [0.0584158415841584, 0.06683168316831682, 3, False],
            # [0.07128712871287128, 0.08217821782178217, 4, False],
        ],
        columns=["lo", "hi", "n_comps", "focal_ad_top"],
    )
    simonov_comp_hq = simonov_comp_hq.melt(
        id_vars=["n_comps", "focal_ad_top"], value_vars=["lo", "hi"]
    )

    return simonov_comp, simonov_comp_hq


def get_est(model, simonov_comp, pooled=False):

    cols = ["n_comps", "focal_ad_top"]
    pred_data = simonov_comp[cols].drop_duplicates()
    pred_data = pred_data.sort_values(by="focal_ad_top")

    if pooled:
        pred_nav = pred_data.copy()
        pred_nav["qry_type"] = "nav"
        pred_brand = pred_data.copy()
        pred_brand["qry_type"] = "brand"
        pred_no_brand = pred_data.copy()
        pred_no_brand["qry_type"] = "not brand"
        pred_data = pd.concat([pred_nav, pred_brand, pred_no_brand])

    est = model.get_prediction(exog=pred_data).summary_frame()
    est = est[["mean_ci_lower", "mean_ci_upper"]]

    est["n_comps"] = pred_data.n_comps.values
    est["focal_ad_top"] = pred_data.focal_ad_top.values
    est = est[est.n_comps <= 3]

    if pooled:
        est["qry_type"] = pred_data.qry_type.values
        id_vars = cols + ["qry_type"]
    else:
        id_vars = cols

    est = est.melt(id_vars=id_vars, value_vars=["mean_ci_lower", "mean_ci_upper"])
    return est


def plot_simonov(models, simonov, outcome):

    simonov["data"] = "Simonov & Hill"

    est = get_est(models[outcome], simonov)
    est["data"] = "Our Data"

    data = pd.concat((est, simonov))
    data = data.rename(
        columns={"focal_ad_top": "Focal Ad Defends", "data": "Data Source"}
    )
    pre_plot()
    sns.lineplot(
        data,
        x="n_comps",
        y="value",
        hue="Data Source",
        style="Focal Ad Defends",
        palette=["#94a4a2", "#832db6"],
    )
    plt.xticks([0, 1, 2, 3])
    plt.ylim(0, 0.25)
    ylabel = (
        "Competitor ad CTR"
        if outcome == "comp_ad_click"
        else "High-Quality Competitor ad CTR"
    )
    plt.gca().set(
        xlabel="# Competitors in Mainline",
        ylabel=ylabel,
    )
    plt.title("Benchmarking Effectiveness")
    plt.gca().legend(loc="upper left", ncol=2, fontsize=9)
    post_plot()


def plot_compare(models, simonov, outcome, focal_ad_top):

    data = get_est(models[outcome], simonov, pooled=True)
    data = data[data.focal_ad_top == focal_ad_top]
    data = data.replace(
        {"nav": "Navigational", "brand": "Brand", "not brand": "Non-Brand"}
    )
    data = data.rename(
        columns={"focal_ad_top": "Focal Ad Defends", "qry_type": "Query Type"}
    )

    pre_plot()
    linestyle = "--" if focal_ad_top else None
    sns.lineplot(
        data,
        x="n_comps",
        y="value",
        hue="Query Type",
        linestyle=linestyle,
    )
    plt.ylim(0, 0.25)
    plt.xticks([0, 1, 2, 3])
    plt.gca().set(
        xlabel="# Competitors in Mainline",
        ylabel="",
    )
    plt.gca().set_yticklabels([])
    defense = "Defense" if focal_ad_top else "No Defense"
    plt.title(f"Comparing Effectiveness: {defense}")
    plt.gca().legend(loc="upper left", ncol=1, fontsize=9)
    post_plot()


def get_simonov_mask(merged):

    # make population as close to Simonov as possible:
    # a) only text ads
    # b) 0 or 1 clicks
    # c) focal ad in ML1 or not on page
    # d) brand appears 80x/day and occupies top ad slot >= 90% of time

    no_missing = merged.no_missing_ad_domains | (merged.ad == 0)
    no_shopping = merged.shopping_ads == 0
    click_mask = (merged.g_ad_clicks + merged.g_organic_clicks) <= 1
    focal_slot = merged.focal_ad_top | ~merged.focal_on_page
    top = merged.top_1500_domain
    mask = no_missing & no_shopping & click_mask & focal_slot & top
    return mask


def add_letter(i):
    b, t = plt.gca().get_ylim()
    l, r = plt.gca().get_xlim()
    plt.gca().text((l + r) / 2, b - 0.08, f"({string.ascii_lowercase[i]})")


def plot_all(models_separate, models_pooled, outcome, simonov, dir_out):
    fig = plt.figure(figsize=(12, 3))
    fig.add_subplot(1, 3, 1)
    plot_simonov(models_separate["nav"], simonov, outcome)
    add_letter(0)
    fig.add_subplot(1, 3, 2)
    plot_compare(models_pooled, simonov, outcome, False)
    add_letter(1)
    fig.add_subplot(1, 3, 3)
    plot_compare(models_pooled, simonov, outcome, True)
    add_letter(2)
    plt.savefig(
        os.path.join(dir_out, f"compare_all_{outcome}.pdf"),
        bbox_inches="tight",
    )


def eval_contrasts(models, dir_out="tables"):

    contrasts = [
        "$beta_6 = 0$",
        "$beta_7 = 0$",
        "$beta_6 + beta_{10} = 0$",
        "$beta_7 + beta_{11} = 0$",
        "$beta_{10} = 0$",
        "$beta_{11} = 0$",
        "$beta_{11} = beta_{10}$",
        "$beta_6 = beta_7$",
    ]

    formulas = [
        "n_comps:C(qry_type, Treatment(reference='nav'))[T.brand] = 0",
        "n_comps:C(qry_type, Treatment(reference='nav'))[T.not brand] = 0",
        "n_comps:C(qry_type, Treatment(reference='nav'))[T.brand] + n_comps:focal_ad_top[T.True]:C(qry_type, Treatment(reference='nav'))[T.brand] = 0",
        "n_comps:C(qry_type, Treatment(reference='nav'))[T.not brand] + n_comps:focal_ad_top[T.True]:C(qry_type, Treatment(reference='nav'))[T.not brand] = 0",
        "n_comps:focal_ad_top[T.True]:C(qry_type, Treatment(reference='nav'))[T.brand]",
        "n_comps:focal_ad_top[T.True]:C(qry_type, Treatment(reference='nav'))[T.not brand]",
        "n_comps:focal_ad_top[T.True]:C(qry_type, Treatment(reference='nav'))[T.not brand] = n_comps:focal_ad_top[T.True]:C(qry_type, Treatment(reference='nav'))[T.brand]",
        "n_comps:C(qry_type, Treatment(reference='nav'))[T.brand] = n_comps:C(qry_type, Treatment(reference='nav'))[T.not brand]",
    ]
    for outcome, model in models.items():
        res = model.t_test(", ".join(formulas), use_t=True)
        adj_pval = stats.multitest.multipletests(res.pvalue, method="fdr_bh")[1]
        data = pd.DataFrame(
            {
                "Observation": ["xyz" for _ in range(len(adj_pval))],
                "Contrast": contrasts,
                "Coef": res.effect.round(3),
                "Std. Err": res.sd.round(3),
                "Adj. Pval": adj_pval.round(3),
            }
        )
        data.to_latex(f"{dir_out}/contrasts_{outcome}.tex", index=False)


def compare_qry_types(merged, dir_out):

    mask = get_simonov_mask(merged)
    simonov, simonov_hq = simonov_baselines()

    models_separate = {
        qry_type: {
            outcome: smf.ols(
                formula=f"{outcome} ~ n_comps*focal_ad_top",
                data=merged[mask & (merged.qry_type == qry_type)],
            ).fit(
                cov_type="cluster",
                cov_kwds={
                    "groups": merged[mask & (merged.qry_type == qry_type)].user_id,
                },
            )
            for outcome in ["comp_ad_click", "comp_ad_click_hq"]
        }
        for qry_type in ["nav", "brand", "not brand"]
    }

    models_pooled = {
        outcome: smf.ols(
            formula=f"{outcome} ~ n_comps*focal_ad_top*C(qry_type, Treatment(reference='nav'))",
            data=merged[mask],
        ).fit(
            cov_type="cluster",
            cov_kwds={
                "groups": merged[mask].user_id,
            },
        )
        for outcome in ["comp_ad_click", "comp_ad_click_hq"]
    }
    eval_contrasts(models_pooled)

    plot_all(models_separate, models_pooled, "comp_ad_click", simonov, dir_out)
    plot_all(models_separate, models_pooled, "comp_ad_click_hq", simonov_hq, dir_out)


def main(dir_out="figures"):

    os.makedirs("figures", exist_ok=True)
    os.makedirs("tables", exist_ok=True)
    merged = pd.read_csv("data/google.csv")
    compare_qry_types(merged, dir_out)


if __name__ == "__main__":
    main()
