# these are all of the signal indicators provided ba finta. For reference:
# https://github.com/peerchemist/finta

# there are all of the finta functions that they currently off taken from their site.
master_finta_map = {
    'Simple Moving Average': 'SMA',
    'Simple Moving Median': 'SMM',
    'Smoothed Simple Moving Average': 'SSMA',
    'Exponential Moving Average': 'EMA',
    'Double Exponential Moving Average': 'DEMA',
    'Triple Exponential Moving Average': 'TEMA',
    'Triangular Moving Average': 'TRIMA',
    'Triple Exponential Moving Average Oscillator': 'TRIX',
    'Volume Adjusted Moving Average': 'VAMA',
    'Kaufman Efficiency Indicator': 'ER',
    'Kaufmans Adaptive Moving Average': 'KAMA',
    'Zero Lag Exponential Moving Average': 'ZLEMA',
    'Weighted Moving Average': 'WMA',
    'Hull Moving Average': 'HMA',
    'Elastic Volume Moving Average': 'EVWMA',
    'Volume Weighted Average Price': 'VWAP',
    'Smoothed Moving Average': 'SMMA',
    'Fractal Adaptive Moving Average': 'FRAMA',
    'Moving Average Convergence Divergence': 'MACD',
    'Percentage Price Oscillator': 'PPO',
    'Volume-Weighted MACD': 'VW_MACD',
    'Elastic-Volume weighted MACD': 'EV_MACD',
    'Market Momentum': 'MOM',
    'Rate-of-Change': 'ROC',
    'Relative Strength Index': 'RSI',
    'Inverse Fisher Transform RSI': 'IFT_RSI',
    'True Range': 'TR',
    'Average True Range': 'ATR',
    'Stop-and-Reverse': 'SAR',
    'Bollinger Bands': 'BBANDS',
    'Bollinger Bands Width': 'BBWIDTH',
    'Momentum Breakout Bands': 'MOBO',
    'Percent B': 'PERCENT_B',
    'Keltner Channels': 'KC',
    'Donchian Channel': 'DO',
    'Directional Movement Indicator': 'DMI',
    'Average Directional Index': 'ADX',
    'Pivot Points': 'PIVOT',
    'Fibonacci Pivot Points': 'PIVOT_FIB',
    'Stochastic Oscillator %K': 'STOCH',
    'Stochastic oscillator %D': 'STOCHD',
    'Stochastic RSI': 'STOCHRSI',
    'Williams %R': 'WILLIAMS',
    'Ultimate Oscillator': 'UO',
    'Awesome Oscillator': 'AO',
    'Mass Index': 'MI',
    'Vortex Indicator': 'VORTEX',
    'Know Sure Thing': 'KST',
    'True Strength Index': 'TSI',
    'Typical Price': 'TP',
    'Accumulation-Distribution Line': 'ADL',
    'Chaikin Oscillator': 'CHAIKIN',
    'Money Flow Index': 'MFI',
    'On Balance Volume': 'OBV',
    'Weighter OBV': 'WOBV',
    'Volume Zone Oscillator': 'VZO',
    'Price Zone Oscillator': 'PZO',
    'Elders Force Index': 'EFI',
    'Cummulative Force Index': 'CFI',
    'Bull power and Bear Power': 'EBBP',
    'Ease of Movement': 'EMV',
    'Commodity Channel Index': 'CCI',
    'Coppock Curve': 'COPP',
    'Buy and Sell Pressure': 'BASP',
    'Normalized BASP': 'BASPN',
    'Chande Momentum Oscillator': 'CMO',
    'Chandelier Exit': 'CHANDELIER',
    'Qstick': 'QSTICK',
    'Twiggs Money Index': 'TMF',
    'Wave Trend Oscillator': 'WTO',
    'Fisher Transform': 'FISH',
    'Ichimoku Cloud': 'ICHIMOKU',
    'Adaptive Price Zone': 'APZ',
    'Squeeze Momentum Indicator': 'SQZMI',
    'Volume Price Trend': 'VPT',
    'Finite Volume Element': 'FVE',
    'Volume Flow Indicator': 'VFI',
    'Moving Standard deviation': 'MSD',
    'Schaff Trend Cycle': 'STC'
}

# these are a bunch of functions that really don't scale well. They probably could work
# with some effort, but running out of time.
bad_funcs = [
    'ADL', 'ADX', 'ATR', 'BBWIDTH', 'BOP', 'CHAIKIN', 'COPP', 'EFI', 'EMV', 'EV_MACD', 
    'IFT_RSI', 'MFI', 'MI', 'MSD', 'OBV', 'PSAR', 'ROC', 'SQZMI', 'STC', 'STOCH', 'ADL', 
    'ADX', 'ATR', 'BBWIDTH', 'BOP', 'CHAIKIN', 'COPP', 'EFI', 'EMV', 'EV_MACD', 'IFT_RSI', 'QSTICK', 
    'MFI', 'MI', 'MSD', 'OBV', 'PSAR', 'ROC', 'SQZMI', 'STOCH', 'UO', 'VORTEX', 'VWAP', 'WTO',
    'WILLIAMS', 'WILLIAMS_FRACTAL', 'ALMA', 'VIDYA','MAMA','LWMA','STOCHD','SWI','EFI']

def getWorkingFunctions():
    """
    Returns a list of working finta functions.

    Returns:
       list of supported finta functions
    """
    working_finta_functions = list(master_finta_map.values())
    for bad_func in bad_funcs:
        if bad_func in working_finta_functions:
            working_finta_functions.remove(bad_func)
    return working_finta_functions


def getFuncsToNamesMap():
    """
    Returns a map of finta functions to their proper names

    Returns:
       dict of finta function names to proper names

    """
    result = {}
    flipped_master = {v: k for k, v in master_finta_map.items()}
    
    working_funcs = getWorkingFunctions()
   
    for func in working_funcs:
        result[func] = flipped_master[func]
    result = {k: v for k, v in sorted(result.items(), key=lambda item: item[1])}
        
    return result

def getNamesToFuncsMap():
    """
    Returns a map of finta proper name to their functions

    Returns:
       dict of finta proper name to their functions
    """
    funcs_to_names = getFuncsToNamesMap()
    return {v: k for k, v in funcs_to_names.items()}        





        
