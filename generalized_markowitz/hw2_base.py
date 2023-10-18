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


def evalfunc(portfolio: np.ndarray, ret: np.ndarray, pi: float, theta: float) -> float:
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


def evalgrad(portfolio: np.ndarray, ret: np.ndarray, pi: float, theta: float) -> np.ndarray:
    """
    Task 1: the objective function gradient


    Parameters
    --------------
    portfolio: np.ndarray: the portfolio vector i.e. x

    ret: np.ndarray: the (T, 3) numpy array containing all asset returns

    pi: float: the exponent parameter of the objective

    theta: the risk-aversion parameter of the objective


    Returns
    --------------
    float: the objective gradient vector
    """
    T = ret.shape[0]
    ret_mu = ret.mean(axis=0)
    delta = ret - ret_mu
    dev = delta.dot(portfolio)
    nom = dev * np.absolute(dev)**(pi-2)
    # p-norm involves abs
    denom = ((np.absolute(dev)**pi).sum())**(1-1/pi)
    return -ret_mu + (
        (theta / T**(1/pi))*nom/denom
    ).dot(delta)


def backtrack(
    x, ret, pi, theta, fval, grad, delta,
    alpha=0.5, beta=0.75
):
    grad_dot_delta = grad.dot(delta)
    step = 1
    goon = True
    success = False
    # print("In backtracking:")
    # print("\talpha =",alpha, "beta =",beta)
    # print("\tgrad_dot_delta =,", grad_dot_delta)
    while goon:
        fnew = evalfunc(x + step * delta, ret, pi, theta)
        target = alpha * step * grad_dot_delta
        # print("\t\ttarget:", target)
        # print("\t\timprovement:", fnew - fval)
        if fnew - fval <= target:
            goon = False
            success = True
        else:
            step *= beta
        if step < 1e-20:
            goon = False
    return step, success


def get_descent(
    x: np.ndarray,
    step: float,
    grad: np.ndarray,
    momentum: bool = False,
    olddelta: np.ndarray = None,
    mu: float = None
) -> np.ndarray:
    if not momentum:
        return - step * grad
    else:
        assert (olddelta is not None) and (mu is not None)
        return -step * grad + (1-mu) * olddelta


def run_grad_desc(
    x: np.ndarray,
    ret: np.ndarray,
    pi: float,
    theta: float,
    x_history: np.ndarray,
    f_history: np.ndarray,
    bt: bool = True,
    bt_a: float = None,
    bt_b: float = None,
    momentum: bool = False,
    mom_mu: float = None,
    max_iter: int = 1000,
    step: float = 0.05,
) -> Tuple[bool, np.ndarray]:
    """
    Task 1: run gradient descent algorithm to compute optimal portfolio vector


    Parameters
    --------------
    x: np.ndarray: portfolio weights

    ret: return data

    pi: float: the exponent parameter of the objective

    theta: float: the risk-aversion parameter of the objective

    x_history: np.ndarray: vector to store portfolio weights

    f_history: np.ndarray: vector to store objective values

    bt: bool: flag for backtracking to obtain adaptive learning rate.

    bt_a: float: the backtracking alpha (acceptance threshold) parameter.

    bt_b: float: the backtracking beta (stepsize decay ratio) parameter.

    momentum: bool: flag for using momentum gradient descent or not.

    mom_mu: float: the momentum memory parameter.

    max_iter: int: iteration limit. Algorithm halts if number of iteration exceeds max_iter.

    step: float: contant step size if backtrack is disabled


    Returns
    --------------
    Tuple[bool, np.ndarray]: a flag for successful convergence and the computed portfolio vector.
    """
    converged = False
    iter = 0
    descent = np.zeros_like(x)
    while iter < max_iter:
        x_history[iter] = x
        fval = evalfunc(x, ret, pi, theta)
        grad = evalgrad(x, ret, pi, theta)
        f_history[iter] = fval
        if bt:
            step, goodstep = backtrack(
                x, ret, pi, theta, fval, grad, -grad, bt_a, bt_b)
        else:
            goodstep = True
        descent = get_descent(x, step, grad, momentum, descent, mom_mu)
        # print(f"iter {iter}, goodstep: {goodstep}")
        if goodstep:
            x += descent
            if (grad * grad).sum() < 1e-8:
                converged = True
                break
            else:
                if iter % 100 == 0:
                    print(f"grad {iter} = {grad}")
        iter += 1
    return iter, converged


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
