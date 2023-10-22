import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from typing import Tuple


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

    portfolio = np.append(portfolio, [1-portfolio.sum()])

    # first part
    drift = -ret_mu.dot(portfolio)

    # second part
    # weighed deviation from mean (part within []^pi)
    deviation = np.abs((ret - ret_mu).dot(portfolio))
    risk = theta * (
        (deviation**pi).mean()
    )**(1/pi)
    return drift + risk

def evalgrad(x, ret, pi, theta):
    T = ret.shape[0]
    ret_mu = ret.mean(axis = 0)
    delta = ret - ret_mu
    ret_mu, ret_mu_c = ret_mu[:-1], ret_mu[-1]
    delta, delta_c = delta[:, :-1], delta[:, -1].reshape(-1, 1)
    dev = (delta - delta_c).dot(x).reshape(-1,1) + delta_c
    nom = dev * np.absolute(dev)**(pi-2)
    denom = ((np.absolute(dev)**pi).sum())**(1-1/pi)
    return -(ret_mu-ret_mu_c) + (
        (theta / T**(1/pi))*nom/denom
    ).T.dot(delta - delta_c).flatten()

def backtrack(
        x, ret, pi, theta, fval, grad, delta,
        alpha = 0.5, beta = 0.75, step_eps=1e-4
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
        if step < step_eps:
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
    step_eps: float = 0.05,
) -> Tuple[bool, np.ndarray]:
    converged = False
    iter = 0
    descent = np.zeros_like(x)
    while iter < max_iter:
        # print("iter:", iter)
        x_history[iter] = x
        fval = evalfunc(x, ret, pi, theta)
        grad = evalgrad(x, ret, pi, theta)
        # print("\tgradient:", grad)
        f_history[iter] = fval
        if bt:
            step, goodstep = backtrack(x, ret, pi, theta, fval, grad, -grad, bt_a, bt_b, step_eps)
        else:
            goodstep = True
            step = step_eps
        # print("\tdecided step:", step)
        descent = get_descent(x, step, grad, momentum, descent, mom_mu)
        # print("\tnew descent:", descent)
        # print(f"iter {iter}, goodstep: {goodstep}")

        # print(descent)
        # print(x)
        if goodstep:
            x += descent
            if np.inner(grad, grad) < 1e-8:
                converged = True
                print("Converged. x:", x)
                break
            else:
                if iter % 10 == 0:
                    print(f"grad {iter} = {grad}")
                    print(f"grad L2 {iter} = {np.inner(grad, grad)}")
        iter += 1
    return iter, converged


def is_open_or_noon(dt: datetime.datetime) -> bool:
    t = dt.time()
    return t == datetime.time(9,30) or t == datetime.time(12,0)

def is_open(dt:datetime.datetime) -> bool:
    return dt.time() == datetime.time(9,30)

def my_dt_parser(s: str) -> datetime.datetime:
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

def read_asset(asset:str, data_dir: str="../data/", return_price: bool=False) -> pd.DataFrame:
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
    df = df.dropna()
    return df.iloc[:T], df.iloc[T:]


if __name__ == "__main__":
    # read
    train, test = read_all()
    ret_train = train.loc[:,[col for col in train.columns if "ret" in col]]
    price_train = train.loc[:,[col for col in train.columns if "price" in col]]

    # run
    ret = ret_train.values
    pi = 2; theta = 10; portfolio = np.array([1/3, 1/3])
    max_iter = 1000
    m = len(portfolio)
    x_history = np.zeros((max_iter, m))
    f_history = np.zeros(max_iter)
    it, converged = run_grad_desc(
        x = portfolio, 
        ret = ret, 
        pi = pi, theta = theta, x_history = x_history, f_history = f_history,
        bt=False, bt_a=0.5, bt_b=0.75, momentum=False, mom_mu=0.8, 
        max_iter = max_iter, step_eps=1e-1, 
    )
    print(f"Converged: {str(converged)}, after {it} iterations")

    # plot
    fig, ax = plt.subplots(1,2, figsize= (15, 5))

    ax[0].plot(f_history[:it])
    portfolio_history = np.hstack([x_history, (1 - x_history.sum(axis = 1)).reshape(-1, 1)])
    ax[1].plot(portfolio_history[:it])

    final_objective = f_history[it-1]
    final_portfolio = portfolio_history[it-1]
    print(f"Final objective = {final_objective}")
    print(f"Final portfolio = {final_portfolio}")

    ax[0].set_title(f"Pi = {pi}, Theta = {theta},\nFinal objective = {final_objective}")
    ax[1].set_title(f"Pi = {pi}, Theta = {theta},\nFinal portfolio = {final_portfolio}")

    