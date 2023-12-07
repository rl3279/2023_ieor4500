import numpy as np
import pandas as pd
import sys
import os

def solveproblem():
    #TODO: Complete me
    pass

def compute_data(datafile = "stockMarketCloseValueswilshire5000.csv"):
    df = pd.read_csv(datafile, low_memory=False)
    df = df.loc[:, [col for col in df.columns if "Close" in col]]
    df = df.dropna(axis=1)
    ret = df.pct_change().dropna()
    print("Return matrix dimensions:", ret.shape)
    cov = ret.cov().values
    mu = ret.mean().values
    np.save("cov.npy", cov)
    np.save("mu.npy", mu)
    return mu, cov

def load_data(
    datafile = "stockMarketCloseValueswilshire5000.csv",
    mufile = "mu.npy", covfile = "cov.npy"
):
    if os.path.exists(mufile) and os.path.exists(covfile):
        return np.load(mufile), np.load(covfile)
    else:
        return compute_data(datafile)
    
def readconfig(log, filename):
    log.joint("reading config file " + filename + "\n")

    try:
        f = open(filename, "r")
    except:
        log.stateandquit("cannot open file", filename)
        sys.exit("failure")
    # read data
    data = f.readlines()
    f.close()

    code = 0
    dataname = 'none'
    thetaval = maxcard = thresh = -1

    for line in data:
        thisline = line.split()
        if thisline[0] == 'datafile':
            dataname = thisline[1]
        elif thisline[0] == 'mufile':
            muname = thisline[1]
        elif thisline[0] == 'covfile':
            covname = thisline[1]
        elif thisline[0] == 'thetaval':
            thetaval = float(thisline[1])
        elif thisline[0] == 'maxcardlong':
            maxcardlong = int(thisline[1])
        elif thisline[0] == 'maxcardshort':
            maxcardshort = int(thisline[1])
        elif thisline[0] == 'thresh':
            thresh = float(thisline[1])
        elif thisline[0] == 'limlong':
            limlong = float(thisline[1])
        elif thisline[0] == 'limshort':
            limshort = float(thisline[1])
        elif thisline[0] == 'END':
            break
        else:
            print('illegal line ' + str(line))
            code = 1
        
        
        #print(thisline[0])

    #print(dataname, thetaval, maxcard, thresh)
    #breakexit('done')

    return (
        code, dataname, muname, covname, thetaval, thresh,
        maxcardlong, maxcardshort, 
        limlong, limshort
    )

    


    

if __name__ == "__main__":
    mu, cov = load_data()
    print(mu)
    print(cov)