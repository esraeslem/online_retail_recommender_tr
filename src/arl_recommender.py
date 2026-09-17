"""
Association Rule Based Recommender System
-------------------------------------------
Miuul Data Scientist Bootcamp — ARL Case Study

Business problem
    Given the basket contents of 3 customers, recommend the most suitable
    product(s) for each one using association rules derived from the
    2010-2011 German customer transactions of the Online Retail II dataset.

Dataset
    Online Retail II — UK-based online retail store's transactions between
    01/12/2009 and 09/12/2011. Mostly gift items, most customers are
    wholesalers. Sheet used: "Year 2010-2011".

Pipeline
    Task 1: Data preparation (drop POST rows, nulls, cancelled invoices,
            non-positive prices; cap outliers in Price & Quantity)
    Task 2: Build the invoice-product matrix and mine association rules
            for German customers
    Task 3: Recommend products for 3 given basket items using the rules

Author: Esra
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)


# ---------------------------------------------------------------------------
# Task 1: Data Preparation
# ---------------------------------------------------------------------------

def outlier_thresholds(dataframe, variable, low_quantile=0.01, up_quantile=0.99):
    """Compute lower/upper outlier thresholds for a numeric variable
    using the given quantiles and the IQR rule."""
    quartile1 = dataframe[variable].quantile(low_quantile)
    quartile3 = dataframe[variable].quantile(up_quantile)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    """Cap (winsorize) values of `variable` that fall outside the
    computed outlier thresholds, in place.

    Note: pandas 3.0 raises a LossySetitemError when assigning a float
    threshold into an int64 column (e.g. Quantity), so the column is
    cast to float64 first.
    """
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe[variable] = dataframe[variable].astype("float64")
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def retail_data_prep(dataframe):
    """Task 1 — Adım 1-6: clean the raw Online Retail II sheet.

    Steps:
        2) Drop StockCode == "POST" (postage fee, not a product)
        3) Drop rows with missing values
        4) Drop cancelled invoices (Invoice containing "C")
        5) Keep rows with Price > 0
        6) Cap outliers in Price and Quantity
    """
    dataframe = dataframe[dataframe["StockCode"] != "POST"]
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].astype(str).str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe


# ---------------------------------------------------------------------------
# Task 2: Association Rules for German Customers
# ---------------------------------------------------------------------------

def create_invoice_product_df(dataframe, id=False):
    """Build the invoice x product pivot table (basket matrix).

    id=True  -> columns are StockCode (product id)
    id=False -> columns are Description (product name)
    Cell values are 1 if the product appears on that invoice, else 0.
    """
    if id:
        pivot = dataframe.groupby(["Invoice", "StockCode"])["Quantity"].sum().unstack()
    else:
        pivot = dataframe.groupby(["Invoice", "Description"])["Quantity"].sum().unstack()
    return pivot.notna().astype(int)


def create_rules(dataframe, id=True, country="Germany"):
    """Task 2 — Adım 2: mine association rules for a given country.

    Filters the dataframe to `country`, builds the basket matrix
    (by product id), runs Apriori, and returns the association rules
    sorted by lift (descending).
    """
    dataframe = dataframe[dataframe["Country"] == country]
    dataframe = create_invoice_product_df(dataframe, id).astype(bool)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    rules = rules.sort_values("lift", ascending=False)
    return rules


# ---------------------------------------------------------------------------
# Task 3: Product Recommendation for Given Basket Items
# ---------------------------------------------------------------------------

def check_id(dataframe, stock_code):
    """Return the unique product name(s) registered for a given StockCode."""
    product_name = dataframe.loc[dataframe["StockCode"] == stock_code, "Description"].unique()
    return product_name


def arl_recommender(rules_df, product_id, rec_count=1):
    """Recommend `rec_count` products for a basket containing `product_id`,
    using rules whose antecedents contain that product, sorted by lift.
    """
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for _, rule in sorted_rules.iterrows():
        for product in rule["antecedents"]:
            if product == product_id:
                recommendation_list.extend(list(rule["consequents"]))

    # de-duplicate while preserving order (in case several rules repeat a product)
    recommendation_list = list(dict.fromkeys(recommendation_list))
    return recommendation_list[:rec_count]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Task 1: read data & prep -----------------------------------------
    df_ = pd.read_excel("data/online_retail_II.xlsx", sheet_name="Year 2010-2011")
    df = df_.copy()
    df = retail_data_prep(df)

    # --- Task 2: German customer association rules -------------------------
    rules = create_rules(df, id=True, country="Germany")
    print(f"Number of association rules found for Germany: {len(rules)}")
    print(rules.head())

    # --- Task 3: recommend products for 3 baskets ---------------------------
    baskets = {
        "User 1": 21987,
        "User 2": 23235,
        "User 3": 22747,
    }

    for user, product_id in baskets.items():
        basket_product_name = check_id(df, product_id)
        recommended_ids = arl_recommender(rules, product_id, rec_count=2)
        recommended_names = [check_id(df, rec_id) for rec_id in recommended_ids]

        print(f"\n{user} — basket item {product_id}: {basket_product_name}")
        print(f"Recommended product id(s): {recommended_ids}")
        for rec_id, rec_name in zip(recommended_ids, recommended_names):
            print(f"  -> {rec_id}: {rec_name}")
