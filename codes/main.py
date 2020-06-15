import numpy as np
import pandas as pd

from Utilities import AddData,sql_select_data

if __name__ == '__main__': 

    # input -- current filename 
    # this includes all the features, and demand. 
    # then fit the model 
    # then make prediction of revenue, given the price. 
    # compute 

    datafile_path = '../../cleaned_data/simulations/'    
    df = sql_select_data('prod1')
    print(df)