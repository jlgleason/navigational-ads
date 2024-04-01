import os
import time
import uuid
import logging
import sys


def setup_logger(fp_parts):
    """setup logger"""
    fp_log = os.path.join(
        "logs",
        f"{fp_parts[-1].split('.')[0]}_{fp_parts[-2]}.log",
    )
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s - %(levelname)s] %(message).5000s",
        filename=fp_log,
        filemode="w",
    )
    logger = logging.getLogger("google.ads.googleads.client")
    logger.setLevel(logging.INFO)
    return logger


def robust_mutate(client, kw_func, operations, logger, wait=5):
    """mutate keyword service, dynamically adjust to API quota limits"""

    while True:
        try:
            return kw_func(customer_id=client.login_customer_id, operations=operations)
        except Exception as e:
            logger.info(e)
            logger.info(f"Retrying in {wait} second(s)...")
            time.sleep(wait)
            wait *= 2


def robust_kw_mutate(client, kw_service, operation, logger, wait=5):
    """mutate keyword plan, dynamically adjust to API quota limits"""

    while True:
        try:
            return kw_service.mutate_keyword_plan_ad_group_keywords(
                customer_id=client.login_customer_id, operations=[operation]
            )
        except Exception as ex:

            try:
                for error in ex.failure.errors:
                    if error.message == "Resource was not found.":
                        logger.info(ex)
                        print(f"RESOURCE NOT FOUND ERROR. EXITING")
                        sys.exit(1)
            except AttributeError:
                continue

            logger.info(ex)
            logger.info(f"Retrying in {wait} second(s)...")
            time.sleep(wait)
            wait *= 2


def robust_request(req_func, req, logger, wait=5):
    """make request to service, dynamically adjust to API quota limits"""

    while True:
        try:
            return req_func(req)
        except Exception as e:
            logger.info(e)
            logger.info(f"Retrying in {wait} second(s)...")
            time.sleep(wait)
            wait *= 2


def keyword_max_len(keywords, max_words=10, max_chars=80):
    """apply max_words and max_chars restrictions"""
    return (
        keywords.apply(lambda q: " ".join(q.split()[:max_words])[:max_chars])
        .drop_duplicates()
        .tolist()
    )


def create_keyword_plan(client, logger):
    """Adds a keyword plan to the given customer account.

    Args:
        client: An initialized instance of GoogleAdsClient

    Returns:
        resource_name: str of the resource_name for the newly created keyword plan.
    """
    keyword_plan_service = client.get_service("KeywordPlanService")
    operation = client.get_type("KeywordPlanOperation")
    keyword_plan = operation.create

    keyword_plan.name = f"Keyword plan for traffic estimate {uuid.uuid4()}"

    forecast_interval = client.enums.KeywordPlanForecastIntervalEnum.NEXT_MONTH
    keyword_plan.forecast_period.date_interval = forecast_interval

    response = robust_mutate(
        client, keyword_plan_service.mutate_keyword_plans, [operation], logger
    )
    resource_name = response.results[0].resource_name
    return resource_name


def create_keyword_plan_campaign(client, keyword_plan, logger):
    """Adds a keyword plan campaign to the given keyword plan.

    Args:
        client: An initialized instance of GoogleAdsClient
        keyword_plan: A str of the keyword plan resource_name this keyword plan
            campaign should be attributed to.create_keyword_plan.

    Returns:
        resource_name: str of the resource_name for the newly created keyword plan campaign.
    """
    keyword_plan_campaign_service = client.get_service("KeywordPlanCampaignService")
    operation = client.get_type("KeywordPlanCampaignOperation")
    keyword_plan_campaign = operation.create

    keyword_plan_campaign.name = f"Keyword plan campaign {uuid.uuid4()}"
    keyword_plan_campaign.cpc_bid_micros = 1_000_000
    keyword_plan_campaign.keyword_plan = keyword_plan

    network = client.enums.KeywordPlanNetworkEnum.GOOGLE_SEARCH
    keyword_plan_campaign.keyword_plan_network = network

    geo_target = client.get_type("KeywordPlanGeoTarget")
    # Constant for U.S. Other geo target constants can be referenced here:
    # https://developers.google.com/google-ads/api/reference/data/geotargets
    geo_target.geo_target_constant = "geoTargetConstants/2840"
    keyword_plan_campaign.geo_targets.append(geo_target)

    # Constant for English
    language = "languageConstants/1000"
    keyword_plan_campaign.language_constants.append(language)

    response = robust_mutate(
        client,
        keyword_plan_campaign_service.mutate_keyword_plan_campaigns,
        [operation],
        logger,
    )
    resource_name = response.results[0].resource_name
    return resource_name


def create_keyword_plan_ad_group(client, keyword_plan_campaign, logger):
    """Adds a keyword plan ad group to the given keyword plan campaign.

    Args:
        client: An initialized instance of GoogleAdsClient
        keyword_plan_campaign: A str of the keyword plan campaign resource_name
            this keyword plan ad group should be attributed to.

    Returns:
        resource_name: str of the resource_name for the newly created keyword plan ad group.
    """
    operation = client.get_type("KeywordPlanAdGroupOperation")
    keyword_plan_ad_group = operation.create

    keyword_plan_ad_group.name = f"Keyword plan ad group {uuid.uuid4()}"
    keyword_plan_ad_group.keyword_plan_campaign = keyword_plan_campaign

    keyword_plan_ad_group_service = client.get_service("KeywordPlanAdGroupService")

    response = robust_mutate(
        client,
        keyword_plan_ad_group_service.mutate_keyword_plan_ad_groups,
        [operation],
        logger,
    )
    resource_name = response.results[0].resource_name
    return resource_name


def create_keyword_plan_ad_group_keywords(
    client, plan_ad_group, keywords, match_type, logger
):
    """Adds keyword plan ad group keywords to the given keyword plan ad group.

    Args:
        client: An initialized instance of GoogleAdsClient
        plan_ad_group: A str of the keyword plan ad group resource_name
            these keyword plan keywords should be attributed to.
        keywords: batch of keywords to add to keyword plan ad group
        match_type: keyword match type

    Returns:
        resource_names: list of KeywordPlanAdGroupKeyword resource names
    """
    keyword_plan_ad_group_keyword_service = client.get_service(
        "KeywordPlanAdGroupKeywordService"
    )

    operations = []
    for kw_text in keywords:
        operation = client.get_type("KeywordPlanAdGroupKeywordOperation")
        kw = operation.create
        kw.text = kw_text
        if match_type == "exact":
            kw.match_type = client.enums.KeywordMatchTypeEnum.EXACT
        elif match_type == "phrase":
            kw.match_type = client.enums.KeywordMatchTypeEnum.PHRASE
        kw.keyword_plan_ad_group = plan_ad_group
        operations.append(operation)

    response = robust_mutate(
        client,
        keyword_plan_ad_group_keyword_service.mutate_keyword_plan_ad_group_keywords,
        operations,
        logger,
    )
    resource_names = [r.resource_name for r in response.results]
    return resource_names


def add_keyword_plan(client, keywords, match_type, logger):
    """Adds a keyword plan, campaign, ad group, etc. to the customer account.

    Args:
        client: An initialized instance of GoogleAdsClient
        keywords: batch of keywords to add to keyword plan ad group
        match_type: keyword match type

    Returns:
        keyword_plan: KeywordPlan:
        keyword_plan_ad_group: KeywordPlanAdGroup
        kw_resources: list of KeywordPlanAdGroupKeyword resource names
    """
    keyword_plan = create_keyword_plan(client, logger)
    keyword_plan_campaign = create_keyword_plan_campaign(client, keyword_plan, logger)
    keyword_plan_ad_group = create_keyword_plan_ad_group(
        client, keyword_plan_campaign, logger
    )
    kw_resources = create_keyword_plan_ad_group_keywords(
        client, keyword_plan_ad_group, keywords, match_type, logger
    )
    return keyword_plan, keyword_plan_ad_group, kw_resources
