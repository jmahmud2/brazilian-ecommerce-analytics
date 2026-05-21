"""
Holdout configuration — PROVIDED (do not modify this file).

The Olist dataset spans roughly September 2016 through August 2018.

Training period: all delivered orders up through TRAIN_END_DATE.
Holdout period: TEST_START_DATE through TEST_END_DATE.
"""

TRAIN_END_DATE = "2018-05-31"
TEST_START_DATE = "2018-06-01"
TEST_END_DATE = "2018-08-31"

# Number of months to forecast (June, July, August 2018)
HOLDOUT_HORIZON = 3