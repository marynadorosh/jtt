import numpy as np


def z_score_normalisation(df):
    """
    Calculate the z score of each columns in the data frame

    Parameters
    ----------
    df : DataFrame
        DataFrame for calculation

    Returns
    -------
    stand_df : DataFrame
        DataFrame with normalized values

    """

    stand_df = (df - df.mean()) / df.std(ddof=0)

    return stand_df


def max_feature_index(df):
    """
    Calculate index of max value for row in DataFrame

    Parameters
    ----------
    df : DataFrame
        DataFrame for calculation

    Returns
    -------
    max_ind_col: Series
        Series with index of max value
    """

    max_ind_col = df.apply(lambda x: np.argmax(x.values), axis=1)

    return max_ind_col


def max_feature_abs_mean_diff(df):
    """
    Calculate difference between of column with max value

    Parameters
    ----------
    df : DataFrame
        DataFrame for calculation

    Returns
    -------
    diff_col: Series
        Series with difference between max value
        and mean of column with max value
    """

    max_values = df.max(axis=1)

    mean_values = df.mean()

    max_ind_values = df.idxmax(axis=1)

    diff_col = max_values - mean_values[max_ind_values].values

    return diff_col
