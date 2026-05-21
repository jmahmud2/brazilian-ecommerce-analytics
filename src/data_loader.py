"""
Data loading utilities — PROVIDED (do not modify this file).

Use these helpers to load the Olist Brazilian E-Commerce dataset.

Usage:
    from data_loader import load_orders, load_order_items
    orders = load_orders()

    # Or load everything at once:
    from data_loader import load_all
    data = load_all()
    orders = data['orders']
"""
import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def load_orders() -> pd.DataFrame:
    return pd.read_csv(
        os.path.join(DATA_DIR, "olist_orders_dataset.csv"),
        parse_dates=[
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )


def load_order_items() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, "olist_order_items_dataset.csv"))


def load_products() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, "olist_products_dataset.csv"))


def load_sellers() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, "olist_sellers_dataset.csv"))


def load_customers() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, "olist_customers_dataset.csv"))


def load_reviews() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, "olist_order_reviews_dataset.csv"))


def load_payments() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, "olist_order_payments_dataset.csv"))


def load_geolocation() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, "olist_geolocation_dataset.csv"))


def load_category_translation() -> pd.DataFrame:
    return pd.read_csv(
        os.path.join(DATA_DIR, "product_category_name_translation.csv")
    )


def load_all() -> dict:
    """Load all datasets and return as a dict keyed by table name."""
    return {
        "orders": load_orders(),
        "order_items": load_order_items(),
        "products": load_products(),
        "sellers": load_sellers(),
        "customers": load_customers(),
        "reviews": load_reviews(),
        "payments": load_payments(),
        "geolocation": load_geolocation(),
        "category_translation": load_category_translation(),
    }