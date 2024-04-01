import sys
import argparse
import json
import os
import time
from datetime import timedelta

import pandas as pd
from google.ads.googleads.client import GoogleAdsClient

from api_utils import (
    create_keyword_plan,
    create_keyword_plan_campaign,
    create_keyword_plan_ad_group,
    robust_kw_mutate,
    robust_request,
    setup_logger,
)


def generate_forecast_curve(client, keyword_plan, logger):
    """makes request to get forecast curve

    Args:
        client: An initialized instance of GoogleAdsClient
        keyword_plan: KeywordPlan instance

    Returns:
        GenerateForecastCurveResponse: forecast curve response
    """
    keyword_plan_service = client.get_service("KeywordPlanService")

    request = client.get_type("GenerateForecastCurveRequest")
    request.keyword_plan = keyword_plan
    return robust_request(keyword_plan_service.generate_forecast_curve, request, logger)


def parse_forecast_curve(response, keyword, dir_output):
    """parse GenerateForecastCurveResponse and write csv

    Args:
        response: GenerateForecastCurveResponse
        keyword: keyword
        dir_output: str representing output directory
    """

    with open(os.path.join(dir_output, f"forecast_curves.json"), "a") as f:
        for curve in response.campaign_forecast_curves:
            for forecast in curve.max_cpc_bid_forecast_curve.max_cpc_bid_forecasts:
                metrics = {
                    "qry": keyword,
                    "max_cpc_bid_micros": forecast.max_cpc_bid_micros,
                    "impressions": forecast.max_cpc_bid_forecast.impressions,
                    "ctr": forecast.max_cpc_bid_forecast.ctr,
                    "average_cpc": forecast.max_cpc_bid_forecast.average_cpc,
                    "clicks": forecast.max_cpc_bid_forecast.clicks,
                    "cost_micros": forecast.max_cpc_bid_forecast.cost_micros,
                }
                f.write(json.dumps(metrics))
                f.write("\n")


def add_keyword(client, kw_plan_ad_group, kw_service, keyword, match_type, logger):
    """adds keyword to keyword plan

    Args:
        client: An initialized instance of GoogleAdsClient
        kw_plan_ad_group: KeywordPlanAdGroup
        kw_service: KeywordPlanAdGroupKeywordService
        keyword: keyword
        match_type: keyword match type

    Returns:
        kw_rsc: keyword resource name
    """

    operation = client.get_type("KeywordPlanAdGroupKeywordOperation")
    kw = operation.create
    kw.text = keyword
    if match_type == "exact":
        kw.match_type = client.enums.KeywordMatchTypeEnum.EXACT
    elif match_type == "phrase":
        kw.match_type = client.enums.KeywordMatchTypeEnum.PHRASE
    kw.keyword_plan_ad_group = kw_plan_ad_group
    response = robust_kw_mutate(client, kw_service, operation, logger)
    kw_rsc = response.results[0].resource_name
    return kw_rsc


def main(client, dir_output, keywords, match_type, start_idx):

    logger = setup_logger(sys.argv[0].split("/"))

    keyword_plan = create_keyword_plan(client, logger)
    keyword_plan_campaign = create_keyword_plan_campaign(client, keyword_plan, logger)
    keyword_plan_ad_group = create_keyword_plan_ad_group(
        client, keyword_plan_campaign, logger
    )

    nb = int(len(keywords) / 1000)
    st = time.time()
    kw_service = client.get_service("KeywordPlanAdGroupKeywordService")
    for i, keyword in enumerate(keywords):

        if i < start_idx:
            continue

        # add keyword
        kw_rsc = add_keyword(
            client, keyword_plan_ad_group, kw_service, keyword, match_type, logger
        )

        # request and parse forecast curve
        response = generate_forecast_curve(client, keyword_plan, logger)
        parse_forecast_curve(response, keyword, dir_output)

        # remove keyword
        operation = client.get_type("KeywordPlanAdGroupKeywordOperation")
        operation.remove = kw_rsc
        robust_kw_mutate(client, kw_service, operation, logger)

        if i % 1000 == 0 and i > 0:
            t = str(timedelta(seconds=time.time() - st))
            print(f"Finished batch {int(i+1)/1000}/{nb}, took {t}")
            st = time.time()


def run(args):

    os.makedirs(args.dir_output, exist_ok=True)
    client = GoogleAdsClient.load_from_storage(f"{args.fp_config}.yaml", version="v11")

    kws = pd.read_csv(args.fp_qrys).iloc[:, 0]
    main(client, args.dir_output, kws, args.match_type, args.start_idx)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Creates batched keyword plans and pulls forecast metrics"
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
        "-m",
        "--match_type",
        type=str,
        required=True,
        help="Whether match type is `exact` or `phrase`.",
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
