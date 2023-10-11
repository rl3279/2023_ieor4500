import numpy as np
import pandas as pd
from typing import Tuple

# Task 1


def read_asset(csv_path: str, return_price: bool) -> pd.DataFrame:
    """
    Task 1: reads a single asset.


    Parameters
    --------------
    csv_path: str: relative path of the csv file containing desired asset.

    return_price: bool: flag to return price along with returns. This is
    needed for test data for computing number of shares to trade at market open.


    Returns
    --------------
    pd.DataFrame: pandas dataframe containing asset returns
    """
    # read csv
    df = pd.read_csv(csv_path)

    # TODO: parse date and time into separate columns

    # TODO: extract only price column

    # TODO: filter out only mkt open and noon ticks of each day

    # TODO: get returns using groupby. At this point date column be a unique identifier

    # TODO: set date to be index

    # TODO: return a dataframe of only returns and date index if
    # return_price is false

    # TODO: if return_price is True, return both the returns df and the price df
    pass


def read_all(data_dir: str, T: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Task 1.5: reads all asset returns and sever into train and test.


    Parameters
    --------------
    T: int: size of the traning period.


    Returns
    --------------
    pd.DataFrame: pandas dataframe containing asset returns
    """
    assets = ["AMZN", "NFLX", "TSLA"]
    # TODO: call read_asset three times to get all three return dfs

    # TODO: inner merge on date index, name each column using asset names

    # TODO: using .iloc to slice the dataframe into train and test

    # TODO: return df.values for train and test
    pass


def objective(portfolio: np.ndarray, ret: np.ndarray, pi: float, theta: float) -> float:
    """
    Task 1: the objective function
    (Remember to vectorize as much as possible)


    Parameters
    --------------
    portfolio: np.ndarray: the portfolio vector i.e. x

    ret: np.ndarray: the (T, 3) numpy array containing all asset returns

    pi: float: the exponent parameter of the objective

    theta: the risk-aversion parameter of the objective


    Returns
    --------------
    float: the objective value.
    """
    # compute mean returns first. ret_mu shape should be (3,)
    ret_mu = ret.mean(axis=0)

    # first part
    drift = -ret_mu.dot(portfolio)

    # second part
    # weighed deviation from mean (part within []^pi)
    deviation = (ret - ret_mu).dot(portfolio)
    risk = theta * (
        (deviation**pi).mean()
    )**(1/pi)
    return drift + risk


def grad(ret: np.ndarray, pi: float, theta: float) -> np.ndarray:
    """
    Task 1: the objective function gradient


    Parameters
    --------------
    ret: np.ndarray: the (T, 3) numpy array containing all asset returns

    pi: float: the exponent parameter of the objective

    theta: the risk-aversion parameter of the objective


    Returns
    --------------
    float: the objective gradient vector
    """
    # TODO: Task for human: compute gradient in vector calculus.

    # TODO: implement aforementioned computation result.
    pass


def step_grad_desc(
    learning_rate: float = 0.05,
    backtrack: bool = True,
    bt_a: float = None,
    bt_b: float = None,
    momentum: bool = False
):
    """
    Task 1: one step of the gradient descent algorithm.
    """
    pass


def run_grad_desc(
    max_iter: int = 1000,
    learning_rate: float = 0.05,
    backtrack: bool = True,
    bt_a: float = None,
    bt_b: float = None,
    momentum: bool = False,
    mom_mu: float = None
) -> Tuple[bool, np.ndarray]:
    """
    Task 1: run gradient descent algorithm to compute optimal portfolio vector


    Parameters
    --------------
    max_iter: int: iteration limit. Algorithm halts if number of iteration exceeds max_iter.

    backtrack: bool: flag for backtracking to obtain adaptive learning rate.

    bt_a: float: the backtracking alpha (acceptance threshold) parameter.

    bt_b: float: the backtracking beta (stepsize decay ratio) parameter.

    momentum: bool: flag for using momentum gradient descent or not.

    mom_mu: float: the momentum memory parameter.


    Returns
    --------------
    Tuple[bool, np.ndarray]: a flag for successful convergence and the computed portfolio vector.
    """
    pass


# Task 2

def validate(ret: np.ndarray, price: np.ndarray, portfolio: np.ndarray) -> float:
    """
    Task 2: validate the portfolio computed in Task 1 by computing portfolio returns.


    Parameters
    --------------
    ret: np.ndarray: the test return data processed in Task 1.

    price: np.ndarray: the market open price data processed in Task 1.

    portfolio: np.ndarray: a (3,) numpy array contaning the output of run_grad_desc.


    Returns
    --------------
    float: the average portfolio return over the test period.
    """

    assert ret.shape[1] == portfolio.shape[0]

    pass
