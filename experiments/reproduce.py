import numpy as np
import scipy
import argparse
import glob
from pathlib import Path
from pathlib import PurePath
import subprocess
import re
from numpy import dtype
import os
from difflib import SequenceMatcher
import time
import json
import sys
import concurrent.futures
import pandas as pd
import argparse
import logging
from logging.handlers import RotatingFileHandler
import psutil


exp_timeout=600
logging.basicConfig(format='%(threadName)s - %(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
    

def getT3SOGAPrograms(programList):
    pfile=open(programList,"r")
    analyzed_programs=pfile.readlines()
    pfile.close()
    
    analyzed_programsName=[ p.strip().split("[")[0].strip() for p in analyzed_programs]
    
    programs=[]
    allprograms=glob.glob("../**/programs/SOGA/**/*.soga",recursive=True)
    for p in allprograms:
        if(Path(p.strip()).name.split(".")[0].strip() in analyzed_programsName ):
            programs.append(Path(p.strip()))
    
    return programs
            
def getT3PSIPrograms(programList):
    pfile=open(programList,"r")
    analyzed_programs=pfile.readlines()
    pfile.close()
    
    analyzed_programsName=[ p.strip().split("[")[0].strip().lower() for p in analyzed_programs]
    
    programs=[]
    allprograms=glob.glob("../**/programs/PSI/**/*.psi",recursive=True)
    for p in allprograms:
        if(Path(p.strip()).name.split(".")[0].strip().lower() in analyzed_programsName ):
            programs.append(Path(p.strip()))
    
    return programs

def getBLOGPrograms(programList):
    pfile=open(programList,"r")
    analyzed_programs=pfile.readlines()
    pfile.close()
    
    analyzed_programsName=[ p.strip().split("[")[0].strip().lower() for p in analyzed_programs]
    
    programs=[]
    allprograms=glob.glob("../**/programs/BLOG/**/*.blog",recursive=True)
    for p in allprograms:
        if(Path(p.strip()).name.split(".")[0].strip().lower() in analyzed_programsName ):
            programs.append(Path(p.strip()))
    
    return programs

def getT3STANPrograms(programList):
    pfile=open(programList,"r")
    analyzed_programs=pfile.readlines()
    pfile.close()
    
    analyzed_programsName=[ p.strip().split("[")[0].strip().lower() for p in analyzed_programs]
    
    programs=[]
    allprograms=glob.glob("../**/programs/STAN/**/*.stan",recursive=True)
    for p in allprograms:
        if(Path(p.strip()).name.split(".")[0].strip().lower() in analyzed_programsName ):
            programs.append(Path(p.strip()))
            
    return programs

def getT3AQUAPrograms(programList):
    pfile=open(programList,"r")
    analyzed_programs=pfile.readlines()
    pfile.close()
    
    analyzed_programsName=[ p.strip().split("[")[0].strip().lower() for p in analyzed_programs]
    
    programs=[]
    allprograms=glob.glob("../**/programs/AQUA/**/*.stan",recursive=True)
    for p in allprograms:
        if(Path(p.strip()).name.split(".")[0].strip().lower() in analyzed_programsName ):
            programs.append(Path(p.strip()))
    
    return programs

def getMatches(matches):
    res=[];
    for matchNum, match in enumerate(matches, start=1):
        #print ("Match {matchNum} was found at {start}-{end}: {match}".format(matchNum = matchNum, start = match.start(), end = match.end(), match = match.group()))
        for groupNum in range(0, len(match.groups())):
            groupNum = groupNum + 1
            #print ("Group {groupNum} found at {start}-{end}: {group}".format(groupNum = groupNum, start = match.start(groupNum), end = match.end(groupNum), group = match.group(groupNum))) 
            res+=match.group(groupNum)
    return "".join(res) 

def runSOGA(program,tvars,parallel=None):
    logger.info(f"solving {program}")
    rt=None
    value=None
    c=None
    d=None
    try:
        if(parallel is None):
            out=subprocess.check_output(["python3","../src/SOGA.py","-f",program],text=True,
                timeout=exp_timeout)
        else:
            out=subprocess.check_output(["python3","../src/SOGA.py","-f",program,"-p",f"{parallel}"],text=True,
                timeout=exp_timeout)
        
        rt_reg = r"\s*Runtime:\s*(\d+\.\d+)"
        c_reg= r"\s*c:\s*(\d+)"
        d_reg= r"\s*d:\s*(\d+)"

        value_reg=None
        value=""
        if(tvars is not None):
            value_reg=r"E\[%s\]: ([\+\-]?\d+.\d+)"%(re.escape(tvars[1].replace('"','').strip()))
            value=getMatches(re.finditer(value_reg, str(out).strip(),  re.MULTILINE | re.IGNORECASE));
         
        rt=getMatches(re.finditer(rt_reg, str(out).strip(), re.IGNORECASE|re.MULTILINE));
        c=getMatches(re.finditer(c_reg, str(out).strip(), re.IGNORECASE|re.MULTILINE));
        d=getMatches(re.finditer(d_reg, str(out).strip(), re.IGNORECASE|re.MULTILINE));

    except subprocess.CalledProcessError as meme:
        value="err"
    except subprocess.TimeoutExpired as toe:
        value="to"
    
    return [rt,value,c,d]


def runAQUA(program,tvars,mean=False):
    print(program)
    stormfiles=glob.glob("%s/**/*.template"%(str(program.parent)),recursive=True)
    rt=-1
    mem=False
    to=False
    value=None
    if(len(stormfiles)==0):
        try:
            st=time.time()
            subprocess.check_call(["java","-cp",
                                     "target/aqua-1.0.jar:lib/storm-1.0.jar",
                                     "aqua.analyses.AnalysisRunner",
                                     "../%s"%(program.parent)],cwd="../tools/AQUA",timeout=exp_timeout,
                                     stdout=subprocess.DEVNULL,
                                     stderr=subprocess.STDOUT)
            rt=time.time()-st
        except subprocess.CalledProcessError as meme:
            mem=True
        except subprocess.TimeoutExpired as toe:
            to=True
    else:
        try:
            st=time.time()
            subprocess.check_call(["java","-cp",
                                     "target/aqua-1.0.jar:lib/storm-1.0.jar",
                                     "aqua.analyses.AnalysisRunner",
                                     "../%s"%(stormfiles[0])],cwd="../tools/AQUA",timeout=exp_timeout,
                                     stdout=subprocess.DEVNULL,
                                     stderr=subprocess.STDOUT)
            rt=time.time()-st
        except subprocess.CalledProcessError as meme:
            mem=True
        except subprocess.TimeoutExpired as toe:
            to=True
    
    if(mem==False and to==False):
        value=None
        if not mean:
            analisys=json.load(open("%s/analysis_%s.txt"%(str(program.parent),tvars[1].replace('"','').strip()),"r"))
            data=np.array(analisys["data"])
            value=np.dot(data[0,:],data[1,:])
        else:
            outFiles=sorted(glob.glob("%s/analysis_*.txt"%(str(program.parent))))
            mu=[]
            for res in outFiles:
                analisys=json.load(open(res,"r"))
                data=np.array(analisys["data"])
                mu+=[np.dot(data[0,:],data[1,:])]
            value=mu[0]*mu[1]+mu[2]
    
    return [rt,value,mem,to]

def runBLOG(program,tvars):
    print(program)
    rt=-1
    mem=False
    to=False
    value=None
    try:
        st=time.time()
        subprocess.check_call([ "./blog",
                                "../../%s"%(program)],cwd="../tools/blog-0.10/bin",timeout=exp_timeout,stdout=subprocess.DEVNULL)
        rt=time.time()-st
    except subprocess.CalledProcessError as meme:
        mem=True
    except subprocess.TimeoutExpired as toe:
        to=True
    
    return [rt,value,mem,to]

def runPSI(program,tvars):
    logger.info(f"Solving {program} with PSI")
    rt=None
    mem=False
    to=False
    value=None
    
    ppath="../%s"%(program)
    
    try:
        st=time.time()
        cwd="../tools/psi"
        psiFormula=subprocess.check_output(["./psi",ppath,"--expectation","--raw","--mathematica"],timeout=exp_timeout,cwd=cwd,text=True)
        #psiFormula=subprocess.check_output(["psisolver",program,"--expectation","--raw","--mathematica"],timeout=exp_timeout,text=True)
        psiFormula="Print[N[%s]]"%(psiFormula)
        logger.info(f"Formula Computed {psiFormula}")

        f=open("results/psi_formula/%s.txt"%(program.name.split(".")[0]),"w+")
        f.write(psiFormula)
        f.close()

        logger.info("Running mathics")
        value=subprocess.check_output(["mathics","-q","-script","./results/psi_formula/%s.txt"%(program.name.split(".")[0])],timeout=exp_timeout,text=True)
        value=str(value).strip()

        if("Syntax::" in value):
            mem=True
            to=True

        rt=time.time()-st
    except subprocess.CalledProcessError as meme:
        mem=True
    except subprocess.TimeoutExpired as toe:
        to=True
    except concurrent.futures._base.TimeoutError as toe2:
        to=True
    
    return [rt,value,mem,to]

def runPYMC3(program,tvars):
    print(program)
    ppath=f"{program.parent.absolute()}/{program.name}"
    outpath=f"{program.parent.absolute()}/{program.stem}.csv"

    rt=None
    mem=False
    to=False
    value=None
    
    cwd=str(Path(program).parent.absolute())
    try:
        st=time.time()
        subprocess.check_call(["python3",ppath,"-o",outpath],timeout=exp_timeout,
                                stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        rt=time.time()-st
    except subprocess.CalledProcessError as meme:
        mem=True
    except subprocess.TimeoutExpired as toe:
        to=True

    if(not to and not mem):
        if(not Path(outpath).is_file()):
            raise valueError(f"{outpath} not found")
        
        data=pd.read_csv(outpath,comment="#")
        value=data["value"].iloc[0]
        rt=data["time"].iloc[0]
        
        os.remove(outpath)

    return [rt,value,mem,to]


def runSTAN(program,tvars,runs=1000,datFile=None):
    ppath="../%s/%s"%(program.parent,program.name.split(".")[0])
    print(program)
    rt=None
    mem=False
    to=False
    value=None

    print(f"Data {datFile}")
    
    cwd="../tools/cmdstan-2.34.0"
    subprocess.check_call(["make",ppath],cwd=cwd,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
    while(True):
        if((Path("%s/%s.data.R"%(str(program.parent),program.name.split(".")[0]))).is_file() or 
            Path(datFile).is_file()):
            try:
                st=time.time()
                if(datFile is None):
                    subprocess.check_call(["%s/%s"%(str(program.parent),program.name.split(".")[0]),"sample","num_samples=%s"%(runs),
                                       "data","file=%s/%s.data.R"%(str(program.parent),program.name.split(".")[0]),"output",
                                       "file=%s.csv"%(program.name.split(".")[0])],timeout=exp_timeout,stdout=subprocess.DEVNULL,
                                                 stderr=subprocess.STDOUT)
                else:
                    subprocess.check_call(["%s/%s"%(str(program.parent),program.name.split(".")[0]),"sample","num_samples=%s"%(runs),
                                       "random","seed=0","data",f"file={datFile}","output",
                                       f"file={program.parent.absolute()}/%s.csv"%(program.name.split(".")[0])],timeout=exp_timeout,stdout=subprocess.DEVNULL,
                                                 stderr=subprocess.STDOUT)

                rt=time.time()-st
            except subprocess.CalledProcessError as meme:
                mem=True
            except subprocess.TimeoutExpired as toe:
                to=True
        else:
            try:
                st=time.time()
                subprocess.check_call(["%s/%s"%(str(program.parent),program.name.split(".")[0]),"sample","num_samples=%s"%(runs)
                                        ,"output", "file=%s.csv"%(program.name.split(".")[0])],timeout=exp_timeout,stdout=subprocess.DEVNULL,
                                             stderr=subprocess.STDOUT)
                rt=time.time()-st
            except subprocess.CalledProcessError as meme:
                mem=True
            except subprocess.TimeoutExpired as toe:
                to=True
        

        if(not to and not mem):
            if(Path(f"{program.parent.absolute()}/%s_out.csv"%(program.name.split(".")[0])).is_file()):
                os.remove(f"{program.parent.absolute()}/%s_out.csv"%(program.name.split(".")[0]))

            subprocess.check_call(["../tools/cmdstan-2.34.0/bin/stansummary",f"{program.parent.absolute()}/%s.csv"%(program.name.split(".")[0]),"-c",
                                   f"{program.parent.absolute()}/%s_out.csv"%(program.name.split(".")[0])],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
            
            data=pd.read_csv(f"{program.parent.absolute()}/%s_out.csv"%(program.name.split(".")[0]),comment="#")
            e=[]
            value=None
            for v in tvars:
                value=data[data["name"]==v.strip().lower()]["Mean"].iloc[0]
                stdDev=data[data["name"]==v.strip().lower()]["StdDev"].iloc[0]
                MCSE=data[data["name"]==v.strip().lower()]["MCSE"].iloc[0]
                ci=1.96*stdDev/np.sqrt(runs)
                rhat=data[data["name"]==v.strip().lower()]["R_hat"].iloc[0]
                #e+=[abs(ci*2)*100/value]
                e+=[MCSE]
                #print(v,value)
            
            os.remove(f"{program.parent.absolute()}/%s.csv"%(program.name.split(".")[0]))
            if(max(e)<=0.1):
                print("converged")
                break
            else:
                print(max(e),runs,rt)
                runs+=1e04
        else:
            print("Timeedout")
            break

    return [rt,value,mem,to]

def saveRes(programs=None,tools=None,outPath=None,tableres=None):
    resFile=open(str(PurePath(outPath)),"w+")
    fileline=""
    for key,val in enumerate(tableres):
        #assumo che la struttura del nome sia tool_model
        exppname=val.replace("Prune","").lower()
        fileline+=f"{exppname}"
        if("soga" not in val.lower()):
            if val in tableres:
                if(tableres[val][2]==True):
                    fileline+=",mem"
                elif(tableres[val][3]==True):
                    fileline+=",to"
                else:
                    fileline+=f",{tableres[val][0]},{tableres[val][1]}"
            else:
                fileline+=",--"
        else:
            if val in tableres:
                fileline+=f",{tableres[val][0]},{tableres[val][1]},{tableres[val][2]},{tableres[val][3]}"
            else:
                fileline+=",--"
        fileline+="\n"
            
    resFile.write(fileline)

    resFile.flush()
    resFile.close()

def Table3():
    print("####################reproducing Table3#####################")
    tvars_soga=np.loadtxt("target_vars_T3.txt",dtype=str,delimiter=",")
    tvars_stan=np.loadtxt("STAN variables.txt",dtype=str,delimiter=",")
    tvars_aqua=np.loadtxt("AQUA variables.txt",dtype=str,delimiter=",")
    tvars_psi=np.loadtxt("PSI variables.txt",dtype=str,delimiter=",")
    
    sogaPrograms=getT3SOGAPrograms("Table3_sogaprograms.txt")
    stanPrograms=getT3STANPrograms("Table3_stanprograms.txt")
    psiPrograms=getT3PSIPrograms("Table3_psiprograms.txt")
    aquaPrograms=getT3AQUAPrograms("Table3_aquaprograms.txt")
    
    tableres={}

    print("####################running SOGA#####################")
    for p in sogaPrograms:
        for idx,var in enumerate(tvars_soga[:,0]):
            if(var.lower()==p.name.split(".")[0].lower()):
                break
        tableres["soga_%s"%(p.name.split(".")[0].replace("Prune","").lower())]=runSOGA(p,tvars_soga[idx,:])
    print("####################running STAN#####################")
    stan_runs=pd.read_csv("STAN_runs.txt")
    for p in stanPrograms:
        runs=stan_runs[stan_runs["program"]==p.name.split(".")[0].strip()]["runs"]
        for idx,var in enumerate(tvars_stan[:,0]):
            if(var.lower()==p.name.split(".")[0].lower()):
                break
        tableres["stan_%s"%(p.name.split(".")[0].lower())]=runSTAN(p,tvars_stan[idx,:],runs.to_numpy()[0])
    print("####################running AQUA#####################")
    for p in aquaPrograms:
        for idx,var in enumerate(tvars_aqua[:,0]):
            if(var.lower()==p.name.split(".")[0].lower()):
                break
        tableres["aqua_%s"%(p.name.split(".")[0].lower())]=runAQUA(p,tvars_aqua[idx,:])
    print("####################running PSI#####################")
    for p in psiPrograms:
        for idx,var in enumerate(tvars_psi[:,0]):
            if(var.lower()==p.name.split(".")[0].lower()):
                break
        tableres["psi_%s"%(p.name.split(".")[0].lower())]=runPSI(p,tvars_psi[idx,:])
    
    pfile=open("Table3_psiprograms.txt","r")
    analyzed_programs=pfile.readlines()
    pfile.close()
    analyzed_programs=[ p.strip().split("[")[0].strip().lower() for p in analyzed_programs]
    
    resFile=open(str(PurePath("./results/Table3.csv")),"w+")
    tools=["STAN","AQUA","PSI","SOGA"]
    
    for p in analyzed_programs:
        fileline=""
        pname=p.lower()
        fileline+=pname
        for t in tools:
            k="%s_%s"%(t.lower(),pname)
            if(t.lower()!="soga"):
                if k in tableres:
                    if(tableres[k][2]==True):
                        if(tableres[k][3]==True):
                            fileline+=",err,-"
                        else:
                            fileline+=",mem,-"
                    elif(tableres[k][3]==True):
                        fileline+=",to,-"
                    else:
                        fileline+=",%s,%s"%(str(tableres[k][0]),str(tableres[k][1]))
                else:
                    fileline+=",--"
            else:
                if k in tableres:
                    fileline+=",%s,%s,%s,%s"%(str(tableres[k][0]),str(tableres[k][1]),str(tableres[k][2]),str(tableres[k][3]))
                else:
                    fileline+=",--"
                
        resFile.write(fileline+"\n")

    resFile.flush()
    resFile.close()

def Table5():
    print("####################reproducing Table5#####################")
    sogaPrograms=getT3SOGAPrograms("Table5_programs.txt")
    psiPrograms=getT3PSIPrograms("Table5_programs.txt")
    blogPrograms=getBLOGPrograms("Table5_programs.txt")

    tableres={}
    print("####################running SOGA#####################")
    for p in sogaPrograms:
        tableres["soga_%s"%(p.name.split(".")[0].replace("Prune","").lower())]=runSOGA(p,None)
    print("####################running PSI#####################")
    for p in psiPrograms:
        tableres["psi_%s"%(p.name.split(".")[0].lower())]=runPSI(p,None)
    print("####################running BLOG#####################")
    for p in blogPrograms:
        tableres["blog_%s"%(p.name.split(".")[0].lower())]=runBLOG(p,None)


    pfile=open("Table5_programs.txt","r")
    analyzed_programs=pfile.readlines()
    pfile.close()
    analyzed_programs=[ p.strip().split("[")[0].strip().lower() for p in analyzed_programs]

    resFile=open(str(PurePath("./results/Table5.csv")),"w+")
    tools=["SOGA","PSI","BLOG"]
    
    for p in analyzed_programs:
        fileline=""
        pname=p.lower()
        fileline+=pname
        for t in tools:
            k="%s_%s"%(t.lower(),pname)
            if(t.lower()!="soga"):
                if k in tableres:
                    if(tableres[k][2]==True):
                        fileline+=",mem"
                    elif(tableres[k][3]==True):
                        fileline+=",to"
                    else:
                        fileline+=",%s"%(str(tableres[k][0]))
                else:
                    fileline+=",--"
            else:
                if k in tableres:
                    fileline+=",%s"%(str(tableres[k][0]))
                else:
                    fileline+=",--"
                
        resFile.write(fileline+"\n")

    resFile.flush()
    resFile.close()

def Table6():
    print("####################reproducing Table6#####################")
    sogaPrograms=getT3SOGAPrograms("Table6_programs.txt")
    stanPrograms=getT3STANPrograms("Table6_programs.txt")
    aquaPrograms=getT3AQUAPrograms("Table6_programs.txt")

    tvars=np.loadtxt("target_vars_T6.txt",dtype=str,delimiter=",")

    tableres={}
    print("####################running SOGA#####################")
    for p in sogaPrograms:
        for idx,var in enumerate(tvars[:,0]):
            if(var.lower()==p.name.split(".")[0].lower()):
                break
        tableres["soga_%s"%(p.name.split(".")[0].replace("Prune","").lower())]=runSOGA(p,tvars[idx,:])
    print("####################running STAN#####################")
    for p in stanPrograms:
        for idx,var in enumerate(tvars[:,0]):
            if(var.lower()==p.name.split(".")[0].lower()):
                break
        tableres["stan_%s"%(p.name.split(".")[0].lower())]=runSTAN(p,tvars[idx,:])
    print("####################running AQUA#####################")
    for p in aquaPrograms:
        for idx,var in enumerate(tvars[:,0]):
            if(var.lower()==p.name.split(".")[0].lower()):
                break
        tableres["aqua_%s"%(p.name.split(".")[0].lower())]=runAQUA(p,tvars[idx,:],True)

    pfile=open("Table6_programs.txt","r")
    analyzed_programs=pfile.readlines()
    pfile.close()
    analyzed_programs=[ p.strip().split("[")[0].strip().lower() for p in analyzed_programs]

    resFile=open(str(PurePath("./results/Table6.csv")),"w+")
    tools=["SOGA","STAN","AQUA"]

    for p in analyzed_programs:
        fileline=""
        pname=p.lower()
        fileline+=pname
        for t in tools:
            k="%s_%s"%(t.lower(),pname)
            if(t.lower()!="soga"):
                if k in tableres:
                    if(tableres[k][2]==True):
                        fileline+=",mem"
                    elif(tableres[k][3]==True):
                        fileline+=",to"
                    else:
                        fileline+=",%s,%s"%(str(tableres[k][0]),str(tableres[k][1]))
                else:
                    fileline+=",--"
            else:
                if k in tableres:
                    fileline+=",%s,%s"%(str(tableres[k][0]),str(tableres[k][1]))
                else:
                    fileline+=",--"
                
        resFile.write(fileline+"\n")

    resFile.flush()
    resFile.close()

def sensPruningExp():
    logger.info("Computing sensisitvity to pruning")
    programs=glob.glob("../**/programs/SOGA/SensitivityExp/Pruning/**/*.soga",recursive=True)
    tvars=np.loadtxt("../programs/SOGA/SensitivityExp/Pruning/observerdVariables.txt",dtype=str,delimiter=",")

    tableres={}
    for p in programs:
        p=Path(p)
        if("NormalMixtures" in p.name):
            continue
        for idx,var in enumerate(tvars[:,0]):
            if(var.lower()==p.name.split(".")[0].lower()):
                break
        pname=p.name.split(".")[0].replace("Prune","").lower()
        expname=f"soga_{pname}_{p.parent.name}"
        tableres[expname]=runSOGA(p,tvars=tvars[idx,:])

    saveRes(programs=programs,tools=["SOGA"],outPath="./results/pruneSensitivity.csv",tableres=tableres)
    
def sensBranchesExp():
    logger.info("Computing sensisitvity to #baranches")
    programs=glob.glob("../**/programs/SOGA/SensitivityExp/#branches/**/*.soga",recursive=True)
    psiPrograms=glob.glob("../**/programs/PSI/SensitivityExp/#branches/**/*.psi",recursive=True)
    tvars=["","x"]

    tableres={}
    logger.info("####################running SOGA#####################")
    for p in programs:
        p=Path(p)
        pname=p.name.split(".")[0].replace("Prune","").lower()
        expname=f"soga_{pname}_{p.parent.name}"
        tableres[expname]=runSOGA(p,tvars=tvars)
    logger.info("####################running PSI#####################")
    for p in psiPrograms:
        p=Path(p)
        pname=p.name.split(".")[0].lower()
        expname=f"psi_{pname}_{p.parent.name}"
        tableres[expname]=runPSI(p,tvars=tvars)

    saveRes(programs=programs,tools=["SOGA","PSI"],
        outPath="./results/branchSensitivity.csv",tableres=tableres)

def sensVarExp():
    logger.info("Computing sensisitvity to variables experiements")
    #programs=glob.glob("../**/programs/SOGA/SensitivityExp/#variables/simplified_ts/*.soga",recursive=True)
    #stanPrograms=glob.glob("../**/programs/SOGA/SensitivityExp/#variables/simplified_ts/STAN/*.stan",recursive=True)
    programs=glob.glob("../**/programs/SOGA/SensitivityExp/#variables/timeseries/*.soga",recursive=True)
    #stanPrograms=glob.glob("../**/programs/STAN/SensitivityExp/#variables/timeseries/*.stan",recursive=True)
    #PYMC3Programs=glob.glob("../**/programs/PYMC/timeseries*.py",recursive=True)

    tableres={}
    # logger.info("####################running STAN#####################")
    # for p in stanPrograms:
    #     p=Path(p)
    #     nvar=int(re.findall(r"(\d+)\.",p.name)[0])
    #     tvars=["alpha","beta"]
    #     tvars+=[f"y{v+1}" for v in range(1,nvar+1)]        
    #     dname=p.name.replace(f"{nvar}","").split(".")[0]
    #     tableres["stan_%s"%(p.name.split(".")[0].lower())]=runSTAN(p,tvars,datFile=f"{p.parent}/{dname}.data.R")
    # logger.info("####################running PyMC3#####################")
    # for p in PYMC3Programs:
    #     p=Path(p)
    #     nvar=int(re.findall(r"(\d+)\.",p.name)[0])
    #     tvars=["alpha","beta"]
    #     tvars+=[f"y{v}" for v in range(1,nvar+1)]        
    #     dname=p.name.replace(f"{nvar}","").split(".")[0]
    #     tableres["pymc_%s"%(p.stem.lower())]=runPYMC3(p,tvars)
    logger.info("####################running SOGA#####################")
    for p in programs:
        p=Path(p)
        nvar=int(re.findall(r"(\d+)\.",p.name)[0])
        tvars=["",f"y{nvar}"]
        tableres["soga_%s"%(p.name.split(".")[0].replace("Prune","").lower())]=runSOGA(p,tvars=tvars)

    resFile=open(str(PurePath("./results/varSensitivity.csv")),"w+")
    tools=["SOGA"]

    for p in programs:
        fileline=""
        p=Path(p)
        pname=p.name.split(".")[0].replace("Prune","").lower()
        fileline+=pname
        for t in tools:
            k="%s_%s"%(t.lower(),pname)
            if(t.lower()!="soga"):
                if k in tableres:
                    if(tableres[k][2]==True):
                        fileline+=",mem"
                    elif(tableres[k][3]==True):
                        fileline+=",to"
                    else:
                        fileline+=",%s,%s"%(str(tableres[k][0]),str(tableres[k][1]))
                else:
                    fileline+=",--"
            else:
                if k in tableres:
                    fileline+=f",{tableres[k][0]},{tableres[k][1]},{tableres[k][2]},{tableres[k][3]}"
                else:
                    fileline+=",--"
                
        resFile.write(fileline+"\n")

    resFile.flush()
    resFile.close()

def sensCmpExp():
    logger.info("Computing sensisitvity component experiements")
    programs=glob.glob("../**/programs/SOGA/SensitivityExp/#components/**/*.soga",recursive=True)
    psiPrograms=glob.glob("../**/programs/PSI/SensitivityExp/#components/**/*.psi",recursive=True)
    tvars=pd.read_csv("target_vars_T3.txt",header=None)
    tvars.iloc[:, 0]=tvars.iloc[:, 0].apply(lambda x:x.lower())

    tableres={}

    logger.info("####################running SOGA#####################")
    for p in programs:
        p=Path(p)
        if("Radar" not in p.name):
            continue
        pname=p.name.split(".")[0].replace("Prune","").lower()
        t=tvars[tvars.iloc[:,0]==pname.replace(re.findall(r"(\d+)",pname)[0],"")].iloc[0,1]
        tableres["soga_%s"%(pname)]=runSOGA(p,tvars=["",t])
    #logger.info("####################running PSI#####################")
    # for p in psiPrograms:
    #     p=Path(p)
    #     pname=p.name.split(".")[0].replace("Prune","").lower()
    #     expname=f"psi_{pname}"
    #     t=tvars[tvars.iloc[:,0]==pname].iloc[0,1]
    #     tableres[expname]=runPSI(p,tvars=["",t])

    
    saveRes(programs=programs,tools=["SOGA"],
        outPath="./results/cmpSensitivity.csv",tableres=tableres)

def sensParExp():
    logger.info("Evaluating Pralellization Experiements")
    programs=glob.glob("../**/programs/SOGA/SensitivityExp/#branches/continuous/RandomWalk/RandomWalk19.soga",recursive=True)

    parLevels=[par-2 for par in range(4,24,2)]

    tableres={}
    logger.info("####################running SOGA#####################")
    tvars=["","x"]
    for p in programs:
        for pl in parLevels:
            p=Path(p)
            nvar=int(re.findall(r"(\d+)\.",p.name)[0])
            tableres[f"soga_%s_{pl}"%(p.name.split(".")[0].replace("Prune","").lower())]=runSOGA(p,tvars=tvars,parallel=pl)

    resFile=open(str(PurePath("./results/parSensitivity.csv")),"w+")
    tools=["SOGA"]

    for p in programs:
        for pl in parLevels:
            fileline=""
            p=Path(p)
            pname=p.name.split(".")[0].replace("Prune","").lower()
            fileline+=f"{pname},{pl}"
            for t in tools:
                k="%s_%s_%s"%(t.lower(),pname,pl)
                if(t.lower()!="soga"):
                    if k in tableres:
                        if(tableres[k][2]==True):
                            fileline+=",mem"
                        elif(tableres[k][3]==True):
                            fileline+=",to"
                        else:
                            fileline+=",%s,%s"%(str(tableres[k][0]),str(tableres[k][1]))
                    else:
                        fileline+=",--"
                else:
                    if k in tableres:
                        fileline+=f",{tableres[k][0]},{tableres[k][1]},{tableres[k][2]},{tableres[k][3]}"
                    else:
                        fileline+=",--"
                    
            resFile.write(fileline+"\n")

    resFile.flush()
    resFile.close()

def monitor_process_and_children(process_id, interval=None):
    """Monitors CPU and memory utilization of the given process and its children."""
    try:
        parent_process = psutil.Process(process_id)
        process_name = parent_process.name()

        # Get CPU utilization (average across cores)
        cpu_percent = parent_process.cpu_percent(interval=interval)

        # Calculate total CPU and memory usage of child processes
        children_cpu = 0
        for child in parent_process.children(recursive=True):
            children_cpu += child.cpu_percent(interval=interval)

        return process_name, cpu_percent, children_cpu

    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
        print(f"Error monitoring process {process_id}: {e}")
        return  None


def main():
    parser = argparse.ArgumentParser(description='SOGA Replication Scripts')
    
    # Adding a string parameter
    parser.add_argument('--exp', required=True,type=str,choices=['t3', 't5', 't6','prune','branch','var','cmp',"par"],
        help='Select the experiement to perform')

    args = parser.parse_args()
    # Accessing the value of the string parameter
    exp = args.exp

    if(exp=="t3"):
        Table3()
    elif(exp=="t5"):
        Table5()
    elif(exp=="t6"):
        Table6()
    elif(exp=="prune"):
        sensPruningExp()
    elif(exp=="branch"):
        sensBranchesExp()
    elif(exp=="var"):
        sensVarExp()
    elif(exp=="cmp"):
        sensCmpExp()
    elif(exp=="par"):
        sensParExp()


    

if __name__ == '__main__':
    main()