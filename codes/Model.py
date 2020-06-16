################################################################################
#
# Model.py       		(c) Cameron Liang 
#						Insight Data Science Fellowship Program 
#     				    cameron.liang@gmail.com
#
# Make and update model fit continuously based on new input data. 
# Make projection based on current best price and compare with original price. 
################################################################################

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

def FindCriticalPrice(prices_changes,revenue_curve):
    """
    Need to make sure that second derivative is negative (i.e., concave down)
    """
    from scipy.interpolate import InterpolatedUnivariateSpline
    f = InterpolatedUnivariateSpline(prices_changes,revenue_curve,k=4)
    cr_pts = f.derivative().roots()
    return cr_pts[0] # assume there is only one local/global maximum 

def ComputeCumulativeRevenue(price,price_changes,demand):
    """
    Parameters:
    ----------
    price: float
        price of product [dollars]
    price_changes: float
        fraction of price changes
    demand: 
        estimates of units ordered
    """
    revenue = np.cumsum((1+price_changes)*price*demand)
    return revenue

def ReadSalesData(filename,product_id):
    df = pd.read_csv(filename)

    df = df.loc[df['product_id'] == product_id]
    time = df['weeks']
    X = df.drop(['weeks','demand','product_id'],axis=1) # remove weeks and demands, select features only
    y = df['demand']
    return time,X,y

def LinearModelFit(X, y, product_id = None, output = None):
    """
    Parameters:
    ----------
    X: array_like
        features that the model depends on; the dependent 
        variables. 
        session: number of unique visitor 
        page_views: number of page views combined all visitors 
        buy_box_perc: percetange on the buybux shared by competitors. 
        price: price in dollars
    y: array
        demand or actual products ordered. 
    
    Returns:
    ----------
    best_price_change: float 
        price change in percentage relative to current price that optimize 
        the revenue
    max_revenue: float
        revenue given the best price change. 
    """
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    
    # make sure slope is negative for price, positive for buybox, etc. 
    model.fit(X,y)
    param = np.concatenate(([model.intercept_],model.coef_))

    # make projection 
    most_recent_features = np.array(X.iloc[-1]) # last row 
    input_delta_prices = X.T.iloc[0]  # get the first column of data - i.e., the price changes. 

    array_size = 100
    prices_changes = np.linspace(-0.8,2,array_size)
    demand_prediction = np.zeros(len(prices_changes))
    for i in range(array_size):
        # note that other parameters might depend on price. so this works only in linear model. 
        # i.e., Buybox(p). Linear model assumes Buybox does not depend on price. 
        features = np.concatenate(([prices_changes[i]],most_recent_features[1:]))
        demand_prediction[i] = model.predict(features.reshape(1,-1))    

    # Compute best price based on current features other than price. 
    # this converts the n-dimensional to 1D model of demand just a function of price. 
    # use feature of the most recent month as an approximate for the next month. 
    revenue_curve = demand_prediction * prices_changes 
    best_price_change = FindCriticalPrice(prices_changes,revenue_curve)

    # note: these demands elasticity is assumed to be independent of time. 
    # what was the baseline ---> why do the demands change in the first place? because i am changing the price. 
    # so the based line of demand is when I don't change the price, given the model. 
    f = interp1d(prices_changes,demand_prediction)
    original_demand = f(0) # zero percent change in price 
    max_rev_demand  = f(best_price_change) # best percent change in price 

    if output == 'demand':
        return f
    else:
        return best_price_change, original_demand, max_rev_demand

def OutputRevenue(time,X,y, original_price, product_id):
    """
    Note that the revenue computed is what it would have been 
    if the price is best up. 
    """
    n_weeks = len(time)
    # accumate arrays of best_demand, original demand, and best_price_change array. 
    best_price_changes = np.zeros(n_weeks)
    original_demand = np.zeros(n_weeks)
    best_demand = np.zeros(n_weeks)

    week_counter = 0 # number 
    for i in range(1,n_weeks): # 198
        week_counter += 1
        temp_x = X.iloc[:i] # all features. 
        temp_y  = y.iloc[:i]

        if i < 10: # do not recommend price change within 10 timesteps to collect data for fit. 
            temp_x = np.array(temp_x)[-1]
            temp_y = np.array(temp_y)[-1]
            best_price_changes[i] =  temp_x[0] # just assume original price. 
            original_demand[i] = temp_y 
            best_demand[i] = temp_y # assume original demand 
        else:
            best_price_changes[i], original_demand[i], best_demand[i] = LinearModelFit(temp_x, temp_y)

    orig_revenue = ComputeCumulativeRevenue(original_price,0.0,original_demand)
    best_revenue = ComputeCumulativeRevenue(original_price,best_price_changes,best_demand)
    
    prediction_data = np.array([time,best_price_changes,orig_revenue,best_revenue])
    df = pd.DataFrame(data=prediction_data.T, columns=['time','best_price_changes','origin_revenue','best_revenue'])
    df['product_id'] = product_id

    output_fname = '../../cleaned_data/simulations/reveune.csv'
    if os.path.isfile(output_fname):
        with open(output_fname,'a') as f: 
            df.to_csv(f,index=False, header=False)
    else:
        df.to_csv(output_fname,index=False)

if __name__ == '__main__':

    import sys
    product_id = sys.argv[1]
    original_price = float(sys.argv[2])
    fname = '../../cleaned_data/simulations/sales.csv'
    main(fname, original_price,product_id)