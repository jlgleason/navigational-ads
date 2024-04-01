import sys
import argparse
import json
import os
import time
from datetime import timedelta
import math

import pandas as pd
from google.ads.googleads.client import GoogleAdsClient

from api_utils import add_keyword_plan, robust_kw_mutate, robust_request, setup_logger


def get_historical_metrics(client, keyword_plan, logger):
    """makes request to get historical metrics

    Args:
        client: An initialized instance of GoogleAdsClient
        keyword_plan: KeywordPlan instance

    Returns:
        GenerateHistoricalMetricsResponse: historical metrics response
    """
    keyword_plan_service = client.get_service("KeywordPlanService")

    agg_metrics = client.get_type("KeywordPlanAggregateMetrics")
    agg_metric = client.enums.KeywordPlanAggregateMetricTypeEnum.DEVICE
    agg_metrics.aggregate_metric_types.append(agg_metric)

    request = client.get_type("GenerateHistoricalMetricsRequest")
    request.keyword_plan = keyword_plan
    request.aggregate_metrics = agg_metrics
    return robust_request(
        keyword_plan_service.generate_historical_metrics, request, logger
    )


def parse_historical_metrics(response, dir_output, idx):
    """parse GenerateHistoricalMetricsResponse and write json

    Args:
        response: GenerateHistoricalMetricsResponse
        dir_output: str representing output directory
        idx: index for batch of keywords
    """

    # After merging: groupby qry_idx, use max device counts
    kwDicts = [
        {
            "avg_monthly_searches": metric.keyword_metrics.avg_monthly_searches,
            "competition": metric.keyword_metrics.competition.name,
            "competition_index": metric.keyword_metrics.competition_index,
            "low_top_of_page_bid": metric.keyword_metrics.low_top_of_page_bid_micros
            / 1_000_000,
            "high_top_of_page_bid": metric.keyword_metrics.high_top_of_page_bid_micros
            / 1_000_000,
            "qry": qry,
            "qry_idx": i,
        }
        for i, metric in enumerate(response.metrics)
        for qry in [metric.search_query] + list(metric.close_variants)
    ]

    with open(os.path.join(dir_output, f"qrys_{idx}.json"), "w") as f:
        json.dump(kwDicts, f)


def get_agg(res):
    """get device metrics from a historical metrics response"""
    return {
        device_searches.device.name: device_searches.search_count
        for device_searches in res.aggregate_metric_results.device_searches
    }


def get_kw_device_breakdown(
    client, kw_plan, kw_service, kw_plan_ad_group, keyword, kw_rsc, agg_total, logger
):
    """triangulate device type breakdown for one query

    Args:
        client: An initialized instance of GoogleAdsClient
        kw_plan: KeywordPlan instance
        kw_service: KeywordPlanAdGroupKeywordService
        kw_ad_group: KeywordPlanAdGroup
        keyword: keyword
        kw_rsc: KeywordPlanAdGroupKeyword resource name
        agg_total: aggregate device type breakdown

    Returns:
        agg_kw: device type breakdown for individual keyword
    """

    # remove keyword and make request
    operation = client.get_type("KeywordPlanAdGroupKeywordOperation")
    operation.remove = kw_rsc
    robust_kw_mutate(client, kw_service, operation, logger)
    response = get_historical_metrics(client, kw_plan, logger)

    # compute device breakdown without keyword i
    agg_no_kw = get_agg(response)
    agg_kw = {k: agg_total[k] - agg_no_kw[k] for k in agg_no_kw.keys()}

    # add keyword i back to plan
    if sum(agg_kw.values()):
        operation = client.get_type("KeywordPlanAdGroupKeywordOperation")
        kw = operation.create
        kw.text = keyword
        kw.match_type = client.enums.KeywordMatchTypeEnum.EXACT
        kw.keyword_plan_ad_group = kw_plan_ad_group
        robust_kw_mutate(client, kw_service, operation, logger)

    return agg_kw


def get_device_breakdown(
    response,
    client,
    kw_plan,
    kw_ad_group,
    kw_resources,
    kw_batch,
    dir_output,
    idx,
    logger,
):
    """triangulate device type breakdown for each query

    Args:
        response: GenerateHistoricalMetricsResponse from request with all queries
        client: An initialized instance of GoogleAdsClient
        kw_plan: KeywordPlan instance
        kw_ad_group: KeywordPlanAdGroup instance
        kw_resources: list of KeywordPlanAdGroupKeyword resource names
        kw_batch: batch of keywords
        dir_output: str representing output directory
        idx: index for batch of keywords
    """

    # overall agg
    agg_total = get_agg(response)
    kw_service = client.get_service("KeywordPlanAdGroupKeywordService")

    all_kws = []
    for keyword, kw_rsc in zip(kw_batch, kw_resources):

        # get individual kw breakdown
        agg_kw = get_kw_device_breakdown(
            client,
            kw_plan,
            kw_service,
            kw_ad_group,
            keyword,
            kw_rsc,
            agg_total,
            logger,
        )
        agg_kw["qry"] = keyword
        all_kws.append(agg_kw)

    with open(os.path.join(dir_output, f"device_{idx}.json"), "w") as f:
        json.dump(all_kws, f)


def main(client, batch_size, dir_output, keywords, start_idx):

    logger = setup_logger(sys.argv[0].split("/"))

    nb = math.ceil(len(keywords) / batch_size)
    start_batch = int(start_idx / batch_size)
    for i in range(nb):

        if i < start_batch:
            continue

        st = time.time()
        kw_batch = keywords[i * batch_size : (i + 1) * batch_size]
        kw_plan, kw_ad_group, kw_rscs = add_keyword_plan(
            client, kw_batch, "exact", logger
        )

        response = get_historical_metrics(client, kw_plan, logger)
        parse_historical_metrics(response, dir_output, i)
        get_device_breakdown(
            response,
            client,
            kw_plan,
            kw_ad_group,
            kw_rscs,
            kw_batch,
            dir_output,
            i,
            logger,
        )

        t = str(timedelta(seconds=time.time() - st))
        print(f"Finished batch {i+1}/{nb}, took {t}")


def run(args):

    os.makedirs(args.dir_output, exist_ok=True)
    client = GoogleAdsClient.load_from_storage(f"{args.fp_config}.yaml", version="v11")

    kws = pd.read_csv(args.fp_qrys).iloc[:, 0]
    main(client, args.batch_size, args.dir_output, kws, args.start_idx)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Creates batched keyword plans and pulls historical metrics"
    )
    parser.add_argument(
        "-f",
        "--fp_config",
        type=str,
        required=True,
        help="Filepath containing Google Ads API config",
    )
    parser.add_argument(
        "-q",
        "--fp_qrys",
        type=str,
        required=True,
        help="Filepath containing Google queries.",
    )
    parser.add_argument(
        "-d",
        "--dir_output",
        type=str,
        required=True,
        help="Directory to output file chunks.",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=1_000,
        help="Maximum number of queries per request.",
    )
    parser.add_argument(
        "-s",
        "--start_idx",
        type=int,
        default=0,
        help="Starts crawling at this query index",
    )
    args = parser.parse_args()
    run(args)
