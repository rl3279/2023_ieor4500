import sys
import csv
import numpy as np
from numpy import linalg as LA
from log import danoLogger
from myutils import breakexit
import time
from versioner import *
from gurobipy import *

def isfloat(test_string):
    try :  
        float(test_string) 
        res = True
    except : 
        res = False
    return res

def solveproblem(log,n, avereturn, cov, thetaval, usebinary, maxcard, thresh):
    log.joint('setting up model\n')

    # model constructor
    themodel = Model("harry")

    # add term to objective function. obj is coefficient. 
    # add mu.dot(x)
    returnvar = themodel.addVar(obj = -1.0, lb = -GRB.INFINITY, ub = GRB.INFINITY, name = 'return')
    # add theta * xQx
    covarvar = themodel.addVar(obj = thetaval, lb = 0, ub = GRB.INFINITY, name = 'covar')

    # xvar dict used to store x variables
    xvar = {}
    # make x bounded by 0 and 1.
    ub = np.zeros(n, dtype = 'd')
    ub.fill(1.0)
    lb = np.zeros(n, dtype = 'd')
    for j in range(n):
        xvar[j] = themodel.addVar(obj = 0.0, lb = lb[j], ub = ub[j], name = 'x_'+str(j))

    # update model
    themodel.update()

    # use lists to contain avg_ret and x's?
    coeff = [avereturn[j] for j in range(n)]
    variables = [xvar[j] for j in range(n)]

    # linear term definition
    expr = LinExpr(coeff, variables)
    themodel.addConstr(expr == returnvar, name='returndef')

    log.joint('setting up quadratic\n')

    # quadratic term definition
    qdef = QuadExpr()
    for i in range(n):
        for j in range(n):
            qdef += cov[i][j]*xvar[i]*xvar[j]

    themodel.addConstr(qdef <= covarvar, name='covardef')

    log.joint('done setting up quadratic\n')

    # add constraint on sum(x) = 1
    coeff = [1.0 for j in range(n)]
    variables = [xvar[j] for j in range(n)]
    expr = LinExpr(coeff, variables)
    themodel.addConstr(expr == 1.0, name='sumx')

    themodel.update()

    if usebinary:
        indvar = {}
        # binary variables
        for j in range(n):
            indvar[j] = themodel.addVar(obj = 0.0, vtype = GRB.BINARY, name = 'i_'+str(j))
        themodel.update()
        # upper bound constraint on x, considering binary case
        for j in range(n):
            themodel.addConstr(xvar[j] <= ub[j]*indvar[j], name='vub_'+str(j))

        # lower bound constraint on x, considering binary case
        if thresh > 0:
            for j in range(n):
                themodel.addConstr(xvar[j] >= thresh*indvar[j], name='vlb_'+str(j))

        # maximum count constraint on the binary variables
        coeff = [1.0 for j in range(n)]
        variables = [indvar[j] for j in range(n)]
        expr = LinExpr(coeff, variables)
        themodel.addConstr(expr <= maxcard, name='card')

        themodel.update()

    # save and write
    lpfilename = 'harry.lp'

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

    log.joint(' xcount: ' + str(count) + '\n')

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
        if thisline[0] == 'file':
            dataname = thisline[1]
        elif thisline[0] == 'thetaval':
            thetaval = float(thisline[1])
        elif thisline[0] == 'maxcard':
            maxcard = int(thisline[1])
        elif thisline[0] == 'thresh':
            thresh = float(thisline[1])
        elif thisline[0] == 'END':
            break
        else:
            print('illegal line ' + str(line))
            code = 1
        
        
        #print(thisline[0])

    #print(dataname, thetaval, maxcard, thresh)
    #breakexit('done')

    return code, dataname, thetaval, maxcard, thresh

    

def getdata(log, filename):
    log.joint("reading file " + filename + "\n")

    try:
        f = open(filename, "r")
    except:
        log.stateandquit("cannot open file", filename)
        sys.exit("failure")

    # read data
    data = f.readlines()
    f.close()        

    line0 = data[0].split()
    n = int(line0[1])
    log.joint(' n = %d\n' %(n))
    avereturn = np.zeros(n)
    cov = {}
    for i in range(n):
        cov[i] = np.zeros(n)

    for j in range(n):
        line = data[j+2].split()
        avereturn[j] = float(line[0])

    position0 = n+2

    print(data[position0])
    position0 += 1
    for i in range(n):
        line = data[position0].split()
        for j in range(n):
            cov[i][j] = line[j]
        #print (position0,i, '*',line)
        position0 += 1

    line = data[position0].split()
    print(data[position0])


    return n, avereturn, cov
    
    
    
if __name__ == '__main__':
    if len(sys.argv) >3 or len(sys.argv)<2:
        print ('Usage: porty.py configfile [logfile]\n')
        exit(0)

    mylogfile = 'port.log'
    if len(sys.argv) == 3:
        mylogfile = sys.argv[4]
        
    log = danoLogger(mylogfile)
    stateversion(log)

    code, dataname, thetaval, maxcard, thresh = readconfig(log, sys.argv[1])


    n, avereturn, cov = getdata(log, dataname)


    usebinary = 0
    if maxcard < n or thresh > 0:
        usebinary = 1

    log.joint(' thetaval = ' + str(thetaval) + ' maxcard = ' + str(maxcard) + ' thresh ' + str(thresh) + '\n')

    breakexit('')

    solveproblem(log,n, avereturn, cov, thetaval, usebinary, maxcard, thresh)

    log.closelog()
