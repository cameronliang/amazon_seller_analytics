import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from Utilities import update_sales,sql_select_data, productIDs
from Utilities import price_product_dict, get_features

from Model import OutputRevenue


def main():
    # input -- current and new sales data file
    datafile_path = '../../cleaned_data/simulations/'
    update_sales(datafile_path)

    # produce cumulative revenue for each product
    product_ids = productIDs()
    price_dict = price_product_dict()
    for pid in product_ids:
        df = sql_select_data(pid)
        time,X,y = get_features(df)
        original_price = price_dict[pid]
        OutputRevenue(time,X,y, original_price, pid)

if __name__ == '__main__': 

    main()
