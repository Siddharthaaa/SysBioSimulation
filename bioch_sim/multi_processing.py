#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:10:00 2019

@author: timur
"""

from threading import Thread
from multiprocessing import Process, Manager, Queue

def run_sims(sims, proc_count = 1):
    
    q1 = Queue()
    q2 = Queue()
    qerr = Queue()
    procs =[]
    for i in range(0,proc_count):
        print("creating process %d" % i )
        p = Process(target = child_process_func, args=(q1,q2,qerr))
        procs.append(p)
        p.start()
        print("Process %d started" % i)
    ids_to_sim = {}
    for sim in sims:
        print("Putting sim %s in the queue" % sim.name)
        q1.put(sim)
        ids_to_sim[id(sim)] = sim
    
    # send end signal to child processes
    for p in procs:
        q1.put(False)
    
    #get results
    try:
        for i in sims:
            res = q2.get(timeout=30)
            sim = ids_to_sim[res["id"]] 
            sim.results = res
            print("Results acquired for %s\n %s" % (sim.name, sim.param_str()))
    except RuntimeError as err:
        print(err)
        print(qerr.get(timeout=1))
    finally:
        [p.terminate() for p in procs]
        [p.join() for p in procs]
    
    

def child_process_func(q1,q2, qerr):
    
    try:
        while True:
            sim = q1.get()
            if sim==False:
                break
            res=sim.simulate()
            res["id"] = sim.id
            q2.put(res)
            del sim
    except RuntimeError as err:
        qerr.put(err)
        
        