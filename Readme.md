This repository contains partial replication code and data for "Search Engine Revenue from Navigational and Brand Advertising", appearing at ICWSM 2024.

### Data

1. **Curlie**: `data/curlie.zip` has 1.3 million entity:URL pairs collected between Feb-Apr 2023 from Curlie: 

<table border="0" bgcolor="#bc001f" cellpadding="3" cellspacing="0">
<tr><td>
<table width="100%" cellpadding="2" cellspacing="0" border="0">
<tr align="center"><td>
<font face="sans-serif, Arial, Helvetica" size="2" color="#ffffff">
Be part of the largest human-edited directory on the web.
</font>
</td></tr>
<tr bgcolor="#f9f9f9" align="center">
<td><font face="sans-serif, Arial, Helvetica" size="2">
<a href="https://curlie.org/public/suggest?cat=$cat">Suggest a Site</a> -
<a href="https:/curlie.org/about.html"><b>Curlie Directory</b></a> -
<a href="https://curlie.org/public/apply?cat=$cat">Become an Editor</a>
</font></td></tr>
</table>
</td></tr>
</table>

### Code

0. **Env**: `pip3 install -r src/requirements.txt`

1. **Ad API Data Collection**: `src/ad_apis` has scripts that collect data from the Google and Bing Ad APIs. These scripts require developer credentials to function, see https://developers.google.com/google-ads/api/docs/start and https://learn.microsoft.com/en-us/advertising/guides. 
    - `src/ad_apis/google/get_historical_metrics.py`: gets Google historical metrics for a set of keywords
    - `src/ad_apis/google/get_forecast_metrics.py`: gets Google forecast metrics for a set of keywords 
    - `src/ad_apis/bing/get_bid_prices.py`: gets Bing historical metrics for a set of keywords

2. **Results Replication**: `data/intermediate.zip` has intermediate measurements corresponding to each SERP. Per Northeastern IRB #20-03-04, we cannot share individual-level data from earlier steps in the analysis pipeline. `src/results` has scripts to reproduce the Section 4 analysis using this intermediate data.
    - `python3 src/results/google_revenue.py run`
    - `python3 src/results/google_revenue.py make_plots`
    - `python3 src/results/microsoft_revenue.py run`
    - `python3 src/results/microsoft_revenue.py make_plots`
    - `python3 src/results/effects.py`
