import os
import json
import logging
import argparse
import sys

import tqdm
from bingads.service_client import ServiceClient
from bingads.authorization import AuthorizationData
import suds
import pandas as pd
import dotenv

from auth_helper import authenticate

dotenv.load_dotenv()

fp_parts = sys.argv[0].split("/")
fp_log = os.path.join(
    "/net/data-ssd/search-ads/logs", f"{fp_parts[-1].split('.')[0]}_{fp_parts[-2]}.log"
)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s - %(levelname)s] %(message).5000s",
    filename=fp_log,
    filemode="w",
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

MAX_CHARS = 100


def get_historical_metrics(authorization_data, kw_batch, match_type, ad_position):
    """make request to GetHistoricalKeywordPerformance operation of AdInsightService

    Args:
        authorization_data: AuthorizationData
        kw_batch: batch of keywords for request
        match_type: Exact or Phrase
        ad_position: MainLine1, MainLine2, MainLine3, or MainLine4
    """

    adinsight_service = ServiceClient(
        service="AdInsightService",
        version=13,
        authorization_data=authorization_data,
        environment=os.environ.get("ENVIRONMENT"),
    )

    keywords = adinsight_service.factory.create("ns1:ArrayOfstring")
    keywords.string.append(kw_batch)

    matchTypes = adinsight_service.factory.create("ArrayOfMatchType")
    matchTypes.MatchType.append([match_type])

    countries = adinsight_service.factory.create("ns1:ArrayOfstring")
    countries.string.append(["US"])

    # needs all device types to align with Google response
    devices = adinsight_service.factory.create("ns1:ArrayOfstring")
    devices.string.append(["Computers", "NonSmartphones", "Smartphones", "Tablets"])

    return adinsight_service.GetHistoricalKeywordPerformance(
        Keywords=keywords,
        TimeInterval="LastMonth",
        TargetAdPosition=ad_position,
        MatchTypes=matchTypes,
        Language="English",
        PublisherCountries=countries,
        Devices=devices,
    )


def build_null_dict(kw):
    return {
        "Keyword": kw,
        "Device": None,
        "MatchType": None,
        "AdPosition": None,
        "Clicks": None,
        "Impressions": None,
        "AverageCPC": None,
        "CTR": None,
        "TotalCost": None,
        "AverageBid": None,
    }


def parse_and_save_response(response, dir_output, ad_position, idx):
    """parse GetHistoricalKeywordPerformance response

    Args:
        response: GetHistoricalKeywordPerformanceResponse
        dir_output: str representing output directory
        ad_position: MainLine1, MainLine2, MainLine3, or MainLine4
        idx: index for batch of keywords
    """

    kws = []
    for kwObject in response["KeywordHistoricalPerformance"]:
        if kwObject.KeywordKPIs is None:
            kwDict = build_null_dict(kwObject.Keyword)
            kws.append(kwDict)
        else:
            for kwKPI in kwObject.KeywordKPIs["KeywordKPI"]:
                kwDict = {"Keyword": kwObject.Keyword} | suds.client.Client.dict(kwKPI)
                kws.append(kwDict)

    logger.info(f"Saving parsed response {idx+1}")
    with open(os.path.join(dir_output, f"qrys_{ad_position}_{idx}.json"), "w") as f:
        json.dump(kws, f, indent=0)


def main(authorization_data, fp_qrys, batch_size, dir_output, match_type):

    os.makedirs(dir_output, exist_ok=True)

    keywords = pd.read_csv(fp_qrys).iloc[:, 0]
    keywords = keywords.apply(lambda q: q[:MAX_CHARS]).tolist()

    ad_positions = [f"MainLine{i}" for i in range(1, 5)]

    nb = int(len(keywords) / batch_size)
    for i in tqdm.tqdm(range(nb)):
        kw_batch = keywords[i * batch_size : (i + 1) * batch_size]
        for ad_position in ad_positions:
            response = get_historical_metrics(
                authorization_data, kw_batch, match_type, ad_position
            )
            parse_and_save_response(response, dir_output, ad_position, i)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Creates batches of keywords and pulls historical metrics"
    )
    parser.add_argument(
        "-q",
        "--fp_qrys",
        type=str,
        required=True,
        help="Filepath containing Bing queries.",
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
        "-m",
        "--match_type",
        type=str,
        required=True,
        help="Whether match type is `Exact` or `Phrase`.",
    )
    args = parser.parse_args()

    logger.info("Loading the web service client proxies...")
    authorization_data = AuthorizationData(
        account_id=None,
        customer_id=None,
        developer_token=os.environ.get("DEVELOPER_TOKEN"),
        authentication=None,
    )
    authenticate(authorization_data)

    main(
        authorization_data,
        args.fp_qrys,
        args.batch_size,
        args.dir_output,
        args.match_type,
    )
