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
import pandas as pd
import matplotlib.pyplot as plt


def LinearModelFit(X, y):
    """
    Parameters:
    ----------
    X: array_like
        shape 
    x: array_like
        features that the model depends on; the dependent 
        variables. 
        session: number of unique visitor 
        page_views: number of page views combined all visitors 
        buy_box_perc: percetange on the buybux shared by competitors. 
        price: price in dollars
    Returns
    ----------
    param: float
        number of units ordered
    """
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X,y)
    param = np.concatenate(([model.intercept_],model.coef_))

    # make projection 
    most_recent_features = np.array(X.iloc[-1]) # last row 
    input_delta_prices = X.T.iloc[0]

    array_size = 100
    n_params = 5
    prices_changes = np.linspace(-0.5,2,array_size)
    demand_prediction = np.zeros(len(prices_changes))
    for i in range(array_size):
        # note that other parameters might depend on price. so this works only in linear model. 
        # i.e., Buybox(p). Linear model assumes Buybox does not depend on price. 
        features = np.concatenate(([prices_changes[i]],most_recent_features[1:]))
        demand_prediction[i] = model.predict(features.reshape(1,-1))    

    # 1. compute best price based on current features other than price. 
    # this converts the n-dimensional to 1D model of demand just a function of price. 
    # use feature of the most recent month as an approximate for the next month. 
    revenue_curve = demand_prediction * prices_changes 
    best_price_change, max_revenue = FindCriticalPrice(prices_changes,revenue_curve)

    return best_price_change, max_revenue

def FindCriticalPrice(x,y):
    from scipy.interpolate import InterpolatedUnivariateSpline
    f = InterpolatedUnivariateSpline(x,y,k=4)
    cr_pts = f.derivative().roots()
    max_revenue = f(cr_pts)
    return cr_pts[0], max_revenue[0]

def ReadSalesData(filename):
    df = pd.read_csv(filename)
    X = df.drop(['weeks','demand'],axis=1) # remove weeks and demands, select features only
    y = df['demand']
    return X,y

def main():
    fname = '../../cleaned_data/simulations/sales.csv'
    X,y = ReadSalesData(fname)
    best_price_change, revenue_change = LinearModelFit(X, y)
    return best_price_change, revenue_change

if __name__ == '__main__':
    
    
    main()