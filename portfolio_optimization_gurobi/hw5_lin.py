import numpy as np 
import numpy.linalg as LA
import pandas as pd 
import time, math, os, sys
from gurobipy import *
from log import danoLogger
from versioner import stateversion
from myutils import breakexit 

## data
def compute_data(datafile = "stockMarketCloseValueswilshire5000.csv"):
    df = pd.read_csv(datafile, low_memory=False)
    df = df.loc[:, [col for col in df.columns if "Close" in col]]
    df = df.dropna(axis=1)
    ret = df.pct_change().dropna()
    print("the return matrix has dimensions:", ret.shape)
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
    
## gurobi optimization
def solveproblem(log, n, avereturn, cov, thetaval, usebinary, L, S, maxcard_pos, maxcard_neg, thresh):
    log.joint('setting up model\n')

    # model constructor
    themodel = Model("hw5")

    # add term to objective function. obj is coefficient. 
    # add mu.dot(x)
    returnvar = themodel.addVar(obj = -1.0, lb = -GRB.INFINITY, ub = GRB.INFINITY, name = 'return')
    # add theta * xQx
    covarvar = themodel.addVar(obj = thetaval, lb = 0, ub = GRB.INFINITY, name = 'covar')

    # dicts used to store x variables
    xvar_pos = {}
    xvar_neg = {}
    # xvar = {}
    # make x_pos, x_neg bounded by 0 and 1.
    ub = np.zeros(n, dtype = 'd')
    ub.fill(1.0)
    lb = np.zeros(n, dtype = 'd')
    for j in range(n):
        xvar_pos[j] = themodel.addVar(obj = 0.0, lb = lb[j], ub = ub[j], name = 'xpos_'+str(j))
        xvar_neg[j] = themodel.addVar(obj = 0.0, lb = lb[j], ub = ub[j], name = 'xneg_'+str(j))
        # xvar[j] = themodel.addVar(obj = xvar_pos[j]-xvar_neg[j], name = "x_"+str(j))
    
    # update model
    themodel.update()

    # use lists to contain avg_ret and x's?
    # coeff = [avereturn[j] for j in range(n)]
    # variables = [xvar[j] for j in range(n)]

    # set up objective
    log.joint('setting up linear\n')

    # linear term definition
    expr = LinExpr([avereturn[j] for j in range(n)], [xvar_pos[j] for j in range(n)]) - LinExpr([avereturn[j] for j in range(n)], [xvar_neg[j] for j in range(n)])
    themodel.addConstr(expr == returnvar, name='returndef')

    log.joint('done setting up linear\n')

    log.joint('setting up quadratic\n')

    # quadratic term definition
    qdef = QuadExpr()
    for i in range(n):
        for j in range(n):
            qdef += cov[i][j]*(xvar_pos[i]-xvar_neg[i])*(xvar_pos[j]-xvar_neg[j])
    themodel.addConstr(qdef <= covarvar, name='covardef')

    log.joint('done setting up quadratic\n')

    # add constraints
    expr_pos = LinExpr([1.0 for j in range(n)], [xvar_pos[j] for j in range(n)])
    themodel.addConstr(expr_pos == L, name='sum_xpos')

    expr_neg = LinExpr([1.0 for j in range(n)], [xvar_neg[j] for j in range(n)])
    themodel.addConstr(expr_neg == S, name='sum_xneg')

    '''
    expr = QuadExpr()
    for j in range(n):
        expr += xvar_pos[j]*xvar_neg[j]
    themodel.addConstr(expr == 0.0, name='zero')
    '''
    yvar_pos = {}
    yvar_neg = {}
    # binary variables
    for j in range(n):
        yvar_pos[j] = themodel.addVar(obj = 0.0, vtype = GRB.BINARY, name = 'y_pos_'+str(j))
        yvar_neg[j] = themodel.addVar(obj = 0.0, vtype = GRB.BINARY, name = 'y_neg_'+str(j))
    themodel.update()

    for j in range(n):
        themodel.addConstr(xvar_pos[j] <= 1-yvar_pos[j], name='x_pos_zero'+str(j))
        themodel.addConstr(xvar_neg[j] <= 1-yvar_neg[j], name='x_neg_zero'+str(j))
        themodel.addConstr(LinExpr([1.0, 1.0], [yvar_pos[j], yvar_neg[j]]) >= 1.0, name='sum_y'+str(j))
    themodel.update()
    

    if usebinary:
        indvar_pos = {}
        indvar_neg = {}
        # binary variables
        for j in range(n):
            indvar_pos[j] = themodel.addVar(obj = 0.0, vtype = GRB.BINARY, name = 'ind_pos_'+str(j))
            indvar_neg[j] = themodel.addVar(obj = 0.0, vtype = GRB.BINARY, name = 'ind_neg_'+str(j))
        themodel.update()

        # upper bound constraint on x, considering binary case
        for j in range(n):
            themodel.addConstr(xvar_pos[j] <= ub[j]*indvar_pos[j], name='vub_pos'+str(j))
            themodel.addConstr(xvar_neg[j] <= ub[j]*indvar_neg[j], name='vub_neg'+str(j))

        # lower bound constraint on x, considering binary case
        if thresh > 0:
            for j in range(n):
                themodel.addConstr(xvar_pos[j] >= thresh*indvar_pos[j], name='vlb_pos'+str(j))
                themodel.addConstr(xvar_neg[j] >= thresh*indvar_neg[j], name='vlb_neg'+str(j))

        # maximum count constraint on the binary variables
        expr_card_pos = LinExpr([1.0 for j in range(n)], [indvar_pos[j] for j in range(n)])
        themodel.addConstr(expr_card_pos <= maxcard_pos, name='card_pos')
        expr_card_neg = LinExpr([1.0 for j in range(n)], [indvar_neg[j] for j in range(n)])
        themodel.addConstr(expr_card_neg <= maxcard_neg, name='card_neg')

        themodel.update()

    # save and write
    lpfilename = 'hw5.lp'

    log.joint("writing to lpfile " + lpfilename + "\n")
    themodel.write(lpfilename)

    # solve
    themodel.optimize()

    log.joint(' optimal solution: return %.4e covar %.4e obj %.4e\n' %(returnvar.x, covarvar.x, thetaval*covarvar.x - returnvar.x))

    # reporting
    count = 0
    for v in themodel.getVars():
        if v.varname != 'covar' and v.varname != 'return' and math.fabs(v.x) > 1e-07:
            if v.varname[0] == 'x':
                count += 1            
                print( v.varname + " = " +str(v.x))
                log.joint( v.varname + " = " +str(v.x) + "\n")

    log.joint(' xcount: ' + str(count) + '\n')

## read config
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
    thetaval = L = S = maxcard_pos = maxcard_neg = thresh = -1

    for line in data:
        thisline = line.split()
        if thisline[0] == 'file':
            dataname = thisline[1]
        elif thisline[0] == 'thetaval':
            thetaval = float(thisline[1])
        elif thisline[0] == 'L':
            L = float(thisline[1])
        elif thisline[0] == 'S':
            S = float(thisline[1])
        elif thisline[0] == 'maxcard_pos':
            maxcard_pos = int(thisline[1])
        elif thisline[0] == 'maxcard_neg':
            maxcard_neg = int(thisline[1])
        elif thisline[0] == 'thresh':
            thresh = float(thisline[1])
        elif thisline[0] == 'END':
            break
        else:
            print('illegal line ' + str(line))
            code = 1

    return code, dataname, thetaval, L, S, maxcard_pos, maxcard_neg, thresh

if __name__ == "__main__":
    if len(sys.argv) >3 or len(sys.argv)<2:
        print ('Usage: porty.py configfile [logfile]\n')
        exit(0)

    mylogfile = 'hw5.log'
    if len(sys.argv) == 3:
        mylogfile = sys.argv[4]
        
    log = danoLogger(mylogfile)
    stateversion(log)

    code, dataname, thetaval, L, S, maxcard_pos, maxcard_neg, thresh = readconfig(log, sys.argv[1])

    mu, cov = load_data(dataname)
    n = len(mu)

    usebinary = 0
    if maxcard_pos < n or maxcard_neg < n or thresh > 0:
        usebinary = 1

    log.joint(f"thetaval = {str(thetaval)}, L = {str(L)}, S = {str(S)}, maxcard_pos = {str(maxcard_pos)}, maxcard_neg = {str(maxcard_neg)}, thresh = {str(thresh)}\n")

    breakexit('')

    solveproblem(log, n, mu, cov, thetaval, usebinary, L, S, maxcard_pos, maxcard_neg, thresh)

    log.closelog()