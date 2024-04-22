# TODO:
# - add instruction for printing on .csv

import random
import numpy as np
import sys
import getopt
import argparse
from multiprocessing import Process,Queue
import sys

from producecfg import *
from libSOGA import *
from sogaPreprocessor import compile2SOGA

from time import time

random.seed(0)
np.random.seed(0)

sys.setrecursionlimit(10000)

def runSoga(cfg,q,useR=False,parallel=None):
    output_dist = None
    output_dist = start_SOGA(cfg,useR=useR,parallel=parallel)
    q.put(output_dist)


def getCliCmd():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="SOGA CLI")

    parser.add_argument("-f","--modelfile", help="SOGA model path",required=True)
    parser.add_argument("-o","--outputfile", help="Output file path",required=False)
    parser.add_argument("-t","--timeout", type=int, default=600,
                        help='The timeout (in seconds) for SOGA computation (default: 600)',required=False)

    # Add optional flag
    parser.add_argument("-c", "--covariance", action="store_true", help="Output covariance",required=False)
    parser.add_argument("-r", "--rmoments", action="store_true", help="Option for computing moments with R package. A running R process is required (default: False)",
        default=False,required=False)
    parser.add_argument("-p", "--parallel", type=int, help="Option for activationg parallelization (default: 1)",default=None,required=False)

    # Add list of strings
    parser.add_argument("-v","--vars", nargs="*", default=[],help="List of output variables",required=False)

    # Parse the command-line arguments
    args = parser.parse_args()

    return args

def printOutput(output_dist,preprocTime,cfgTime,sogaTime,args):
    var_list = []
    var_idx = []

    if args.outputfile is not None:
        sys.stdout=open(args.outputfile, 'w')

    if(output_dist is not None):

        mwidth=np.max([len(preprocTime),len(cfgTime),len(sogaTime)])

        print(f'SOGA preprocessing in: {preprocTime.rjust(mwidth)} s')
        print(f'      CFG produced in: {cfgTime.rjust(mwidth)} s')
        print(f'              Runtime: {sogaTime.rjust(mwidth)} s')
        print(f"c: {(output_dist.gm.n_comp()):d}")
        print(f"d: {(len(output_dist.var_list)):d}")

        if len(args.vars) == 0:
            for var, val in zip(output_dist.var_list, output_dist.gm.mean()):
                print('E['+var+']:', round(val,5))
        else:
            for var in args.vars:
                i = output_dist.var_list.index(var)
                print('E['+var+']:', round(output_dist.gm.mean()[i],5))
                var_idx.append(i)
        print('\n')

        if args.covariance:
            if len(args.vars) == 0:
                print('Covariance:\n', np.around(output_dist.gm.cov(),5))
            else:
                print('Covariance:\n', np.around(output_dist.gm.cov()[var_idx][:,var_idx], 5))
    else:
        print("SOGA Timedout")

def printBanner():
    print('/ ___| / _ \ / ___|  / \\\n\___ \| | | | |  _  / _ \\\n ___) | |_| | |_| |/ ___ \\\n|____/ \___/ \____/_/   \_\\\n')


def SOGA():
    printBanner()

    args=getCliCmd()
    preproc_strt=time()
    compiledFile=compile2SOGA(args.modelfile)
    preproc_end=time()
    
    cfg_start = time()
    cfg = produce_cfg(compiledFile)
    cfg_end = time()

    comp_start = time()
    #output_dist = start_SOGA(cfg)
    #q = Queue()
    #sogaProcess = Process(target=runSoga, args=(cfg,q,args.rmoments,args.parallel))
    # Start the thread
    #sogaProcess.start()
    # Wait for the process to finish 
    #output_dist=None

    output_dist = start_SOGA(cfg,useR=useR,parallel=parallel)
    try:
        output_dist=q.get(timeout=args.timeout)
    except Exception as e:
        pass
    finally:
        sogaProcess.terminate()
        comp_end = time()

    preprocTime=f"{preproc_end-preproc_strt:<.3f}"
    cfgTime=f"{cfg_end-cfg_start:<.3f}"
    sogaTime=f"{comp_end-comp_start:<.3f}"

    printOutput(output_dist=output_dist,preprocTime=preprocTime
        ,cfgTime=cfgTime,sogaTime=sogaTime,args=args)


if __name__ == '__main__':
    SOGA()    