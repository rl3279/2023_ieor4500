import datetime
import numpy as np
import pandas as pd
from typing import Tuple

# Task 1


def is_open_or_noon(dt: datetime.datetime) -> bool:
    t = dt.time()
    return t == datetime.time(9,30) or t == datetime.time(12,0)

def is_open(dt:datetime.datetime) -> bool:
    return dt.time() == datetime.time(9,30)

def my_dt_parser(s: str) -> datetime.datetime:
    """
    parser for IEOR4500 intraday tick price files.

    This is 6 times faster than pd.DatetimeIndex.
    """
    date, time = s.split()
    m, d, y = date.split("/")
    H, M = time.split(":")
    return datetime.datetime(
        year = 2000 + int(y),
        month = int(m),
        day = int(d),
        hour = int(H),
        minute = int(M)
    )

def read_asset(asset:str, data_dir: str="../data/") -> pd.DataFrame:
    """
    Task 1: reads a single asset.


    Parameters
    --------------
    asset: str: asset name e.g. "AMZN"

    data_dir: str: your local folder contaning .csv files

    Returns
    --------------
    pd.DataFrame: pandas dataframe containing asset returns
    """
    # read csv
    csv_path = data_dir + asset + ".csv"
    df = pd.read_csv(csv_path, header=3).loc[:, ["Dates", "Close"]]
    # read up to empty entries
    df = df.iloc[:df["Close"].isna().argmax()]
    
    # manual hard-coding processing
    if asset == "NFLX":
        df.loc[0, "Dates"] = "2/1/21 9:30"
    elif asset == "AMZN":
        df.loc[0, "Dates"] = "1/4/21 9:30"
        # two missing Close on 2021-04-20 and 2021-06-14. Backfill using 12:01 data
        missing = ["4/20/21 12:00", "6/14/21 12:00"]
        df = pd.concat([df, pd.DataFrame([[m, np.nan] for m in missing], columns = df.columns)], ignore_index=True)
        df.sort_values(by = "Dates")
        df.fillna(method="backfill")

    # extract open or noon data
    df["dt"] = df["Dates"].apply(my_dt_parser)
    df["Date"] = df["dt"].apply(lambda dt: dt.date())
    open_or_noon = df["dt"].apply(is_open_or_noon)
    df = df.loc[open_or_noon]

    # compute daily return
    ret = df.loc[:, ["Close","Date"]].groupby("Date").pct_change().values
    ret = ret[~np.isnan(ret)]

    # return along with daily open price
    df = df.loc[df["dt"].apply(is_open)]
    df["ret"] = ret
    df.set_index("Date", inplace=True)
    df = df[["ret", "Close"]]
    df.rename(columns = {"ret": f"{asset}_ret", "Close": f"{asset}_price"}, inplace = True)
    return df


def read_all(data_dir: str = "../data/", T: int = 100) -> Tuple[np.ndarray, np.ndarray]:
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

    dfs = [read_asset(asset, data_dir) for asset in assets]

    df = dfs[0]
    for i in range(1, 3):
        df = df.join(dfs[i])

    return df.dropna()


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
