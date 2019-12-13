#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 14:33:10 2019

@author: timur
"""

from . import SimParam

def get_exmpl_sim(name = ("basic", "LotkaVolterra", "hill_fb")):
    s = None
    if(type(name) != str):
        name = "basic"
    if(name == "basic"):
        s1= s2 = s3 = d1 = d2 = d3 = d0 = 1
        d1 = 0.1
        d2 = 0.1
        k_syn = 100
        
        name="Basic"
        s = SimParam(name,100, 1001,
                     params = {"v_syn": k_syn, "s1": s1, "s2": s2, "s3": s3, "d0": d0, "d1": d1, "d2": d2, "d3": d3},
                     init_state = {"pre_RNA": 1, "Incl": 1, "Skip":1, "ret": 1})
        
        s.simulate_ODE = True
        
        s.add_reaction("v_syn", {"pre_RNA":1}, "Transcription" )
        s.add_reaction("d0*pre_RNA", {"pre_RNA":-1}, "mRNA degr." )
        s.add_reaction("s1*pre_RNA", {"pre_RNA":-1, "Incl":1}, "Inclusion")
        s.add_reaction("d1*Incl", {"Incl": -1} , "Incl. degr." )
        s.add_reaction("s2*pre_RNA" ,  {"pre_RNA":-1, "Skip":1}, "Skipping")
        s.add_reaction("d2*Skip", {"Skip":-1}, "Skip. degr."  )
        s.add_reaction("s3*pre_RNA", {"pre_RNA":-1, "ret":1}, "Retention" )
        s.add_reaction("d3*ret",  {"ret": -1}, "ret. degr" )
    elif(name == "test"):
        params = {"p1":1, "p2":2, "p3":3, "degr":100}
        s = SimParam(name, 100,1001, params,
                     init_state = {"A1": 10, "A2": 15, "A3": None})
        s.simulate_ODE = False
        s.add_reaction("p1*A1", {"A1":-1, "A3":2})
        s.add_reaction("p2*A2", {"A1":1, "A3":1})
        s.add_reaction("p3*A3", {"A3":-1, "A1":"A3*p2"})
        s.add_reaction("degr*A1", {"A1":-1 })
        s.add_reaction("degr*A3", {"A3":"-2*A1" })
        s.add_reaction("degr*A2", {"A2":-1 })
        
    elif(name == "hill_fb"):
                
        k_on=0
        k_off=0.00
        v_syn=10.000
        s1=0.500
        Ka1=70
        n1=6.000
        s2=1.000
        s3=0.100
        d0=0.100
        d1=0.100
        d2=0.100
        d3=0.500
        s1_t=5
        name="Hill_Feedback"
        s = SimParam(name, 400, 1001,
                     params = {"k_on": k_on, "k_off": k_off, "v_syn": v_syn, "s1": s1, "Ka1":Ka1, "n1":n1,
                               "s2": s2, "s3": s3, "d0": d0, "d1": d1, "d2": d2, "d3": d3, "s1_t":s1_t},
                     init_state = {"Pr_on": 1, "Pr_off": 0, "pre_RNA": 0,
                                   "Incl": 10, "Skip": 0, "ret": 0})
        
        s.simulate_ODE = True
        
        s.add_reaction("k_on*Pr_off", {"Pr_on":1, "Pr_off":-1}, "Prom. act.")
        s.add_reaction("k_off*Pr_on", {"Pr_off":1, "Pr_on":-1}, "Prom. deact.")
        s.add_reaction("v_syn*Pr_on", {"pre_RNA":1, "Pr_on":[-1,1]}, "Transcription")
        s.add_reaction("d0*pre_RNA", {"pre_RNA":-1}, "mRNA degr." )
        s.add_reaction("s1*pre_RNA + pre_RNA* s1_t * (1/(1 + (Ka1/Incl)**n1) if Incl > 0 else 0)",
                       {"pre_RNA":-1, "Incl":1}, "Inclusion")
        s.add_reaction("d1*Incl", {"Incl": -1}, "Incl. degr"  )
        s.add_reaction("s2*pre_RNA" ,  {"pre_RNA":-1, "Skip":1}, "Skipping")
        #            s.add_reaction("s2*pre_RNA + 5* (Skip > 0)* 1/(1+(Ka2/Skip)**n2)" ,  {"pre_RNA":-1, "Skip":1})
        s.add_reaction("d2*Skip", {"Skip":-1}, "Skip degr."  )
        s.add_reaction("s3*pre_RNA", {"pre_RNA":-1, "ret":1}, "Retention" )
        s.add_reaction("d3*ret",  {"ret": -1}, "Ret. degr." )
        
        s.set_cluster("Pr_on", (1,))
        s.set_cluster("Pr_off", (1,))
        
    elif(name == "LotkaVolterra"):
        s = SimParam("Lotka Volterra", 50, 301,
                      {"k1":1, "k2":0.007, "k3":0.6 },
                      {"Prey":50, "Predator":200})
        s.add_reaction("k1*Prey", {"Prey":[-2,3]}, "Reproduction")
        s.add_reaction("k2*Prey*Predator",{"Prey":-1, "Predator":1}, "Hunt")
        s.add_reaction("k3*(Predator)",{"Predator":-1}, "Death")
    elif(name == "CoTrSplicing"):
        
        # https://www.ncbi.nlm.nih.gov/pubmed/15217358
        gene_len = 3000
        u1_1_bs_pos = 150
        u2_1_bs_pos = 1500
        u1_2_bs_pos = 1700
        u2_2_bs_pos = 2800
        
        v0 = 60
        spl_r = 0.8
        
        v1_1 = 0.234
        v2_1 = 0.024
        v1_2 = 0.012
        v2_2 = 1.25
        s2 = 1/(3/(v1_1+v2_2)  + 1/(spl_r))
        s1 = 1/(3/(v1_1+v2_1)  + 3/(v1_2+v2_2) + 2/(spl_r))
        
        # consider https://science.sciencemag.org/content/sci/331/6022/1289/F5.large.jpg?width=800&height=600&carousel=1
        #for Ux binding rates
        params = {"pr_on": 2, "pr_off" : 0.1,
                "elong_v": v0, # 20-80 nt per second . http://book.bionumbers.org/what-is-faster-transcription-or-translation/
                "gene_len": gene_len,
                "spl_rate": spl_r,#0.002, # 1/k3 = 1/k2 + 1/k1
                "u1_1_bs_pos": u1_1_bs_pos , # U1 binding site position
                "u1_2_bs_pos": u1_2_bs_pos ,
                "u2_1_bs_pos": u2_1_bs_pos, # U2 bind. site pos 1
                "u2_2_bs_pos": u2_2_bs_pos,
                "u1_1_br": v1_1,  # binding rate of U1 
                "u2_1_br": v2_1,
                "u1_2_br": v1_2,  # binding rate of U1
                "u2_2_br": v2_2,
                "u1ur": 0.001,  # unbinding rate of U1 
                "u2ur": 0.001, # unbinding rate of U1
#                "tr_term_rate": 100,
#                "s1":s1, "s2":s2, "s3": 1e-4,
                # http://book.bionumbers.org/how-fast-do-rnas-and-proteins-degrade/
                "d1": 2e-4, "d2":2e-4, "d3":1e-3 # mRNA half life: 10-20 h -> lambda: math.log(2)/hl
                }
        
        
        s = SimParam("Cotranscriptional splicing", 10000, 10001, params = params,
                        init_state = {"Pol_on":0, "Pol_off": 1,
                                      "nascRNA_bc": 0,
                                      "Pol_pos": 0,
                                      "Skip":0, "Incl": 0, "ret": 0,
                                      "U1_1":0, "U1_2":0,   #Splicosome units U1 binded
                                      "U2_1":0, "U2_2":0,
                                      "Intr1":0, "Intr2":0, "Exon1":0,
                                      "U11p":0, "U21p":0, "U12p":0, "U22p":0})
        # for drawing PN with SNAKES
        [s.set_cluster(sp,(0,)) for sp in ["Incl", "Skip"]]
        [s.set_cluster(sp,(1,)) for sp in ["ret", "ret_i1", "ret_i2"]]
        [s.set_cluster(sp,(2,)) for sp in ["Intr1", "Intr2", "Exon1"]]
        [s.set_cluster(sp,(3,)) for sp in ["Intr1_ex", "Intr2_ex", "Exon1_ex"]]
        [s.set_cluster(sp,(4,)) for sp in ["Pol_on", "Pol_off", "Pol_pos"]]
        [s.set_cluster(sp,(5,)) for sp in ["U1_1", "U1_2", "U2_1", "U2_2"]]
        ###########################
        
        s.add_reaction("pr_on * Pol_off",
                       {"Pol_on":1, "Pol_off": -1,"Exon1":1, "Intr1":1,"Intr2":1,
                        "Tr_dist": "gene_len", "nascRNA_bc": "-nascRNA_bc"},
                       name = "Transc. initiation")
        
        s.add_reaction("elong_v * Pol_on",
                       {"nascRNA_bc": 1, "Pol_pos":1, "Tr_dist":-1},
                       name = "Elongation")
        
        # Ux (un)binding cinetics
        s.add_reaction("u1_1_br * Intr1",
                       {"U1_1":[1,None], "Intr1": [-1,1],
                        "Pol_pos":["-u1_1_bs_pos", "u1_1_bs_pos"]},
                       "U1_1 binding")
        s.add_reaction("u1ur * U1_1", {"U1_1":-1}, "U1_1 diss.")
        
        s.add_reaction("u1_2_br * Intr2",
                       {"U1_2":[1,None], "Intr2": [-1,1],
                        "Pol_pos":["-u1_2_bs_pos", "u1_2_bs_pos"]},
                        "U1_2 binding")
        s.add_reaction("u1ur * U1_2", {"U1_2":-1}, "U1_2 diss.")
        
        s.add_reaction("u2_1_br * Intr1",
                      {"U2_1":[1,None], "Intr1": [-1,1],
                        "Pol_pos":["-u2_1_bs_pos", "u2_1_bs_pos"]},
                       "U2_1 binding")
        s.add_reaction("u2ur * U2_1", {"U2_1":-1}, "U2_1 diss.")
        
        s.add_reaction("u2_2_br * Intr2",
                       {"U2_2":[1,None], "Intr2": [-1,1],
                        "Pol_pos":["-u2_2_bs_pos", "u2_2_bs_pos"]},
                        "U2_2 binding")
        s.add_reaction("u2ur * U2_2", {"U2_2":-1}, "U2_2 diss.")
        
        #Splicing
        s.add_reaction("U1_1 * U2_1 * Intr1 * spl_rate",
                       {"Intr1":-1, "U1_1":-1, "U2_1":-1,
                        "nascRNA_bc": "-(u2_1_bs_pos - u1_1_bs_pos)"},
                       name="Intron 1 excision")
        s.add_reaction("U1_2 * U2_2 * Intr2 * spl_rate",
                       {"Intr2":-1, "U1_2":-1, "U2_2":-1,
                        "nascRNA_bc": "-(u2_2_bs_pos - u1_2_bs_pos)"},
                       name="Intron 2 excision")
        s.add_reaction("U1_1 * U2_2 * spl_rate",
                       {"Intr1":-1, "Intr2":-1, "Exon1":-1,
                        "nascRNA_bc": "-(u2_2_bs_pos - u1_1_bs_pos)",
                        "U1_1":-1, "U2_2":-1, "U1_2":"-U1_2", "U2_1":"-U2_1"},
                       name="Exon 1 excision (inclusion)")
        
        #Transcription termination
        s.add_reaction("elong_v",
                       {"Intr1":None, "Intr2":None, "Exon1":None,
                        "Skip":1, "Pol_pos": "-gene_len", "Pol_on":-1, "Pol_off":1},
                       name = "Termination: skipping")
        s.add_reaction("elong_v",
                       {"Intr1":None, "Intr2":None, "Exon1":-1, "Incl":1,
                        "Pol_pos": "-gene_len", "Pol_on":-1, "Pol_off":1},
                       name = "Termination: inclusion")
        s.add_reaction("elong_v",
                       {"Exon1":-1, "Intr1":-1, "Intr2":None, "ret_i1":1,
                        "Pol_pos": "-gene_len", "Pol_on":-1, "Pol_off":1,
                        "U1_1":"-U1_1", "U2_1": "-U2_1",
                        "U11p":"U1_1", "U21p":"U2_1"},
                       name = "Termination: ret i1")
        s.add_reaction("elong_v",
                       {"Exon1":-1, "Intr2":-1, "Intr1":None, "ret_i2":1,
                        "Pol_pos": "-gene_len", "Pol_on":-1, "Pol_off":1,
                        "U1_2":"-U1_2", "U2_2": "-U2_2",
                        "U12p":"U1_2", "U22p":"U2_2"},
                       name = "Termination: ret i2")
        s.add_reaction("elong_v",
                       {"Exon1":-1, "Intr1":-1, "Intr2":-1, "ret":1,
                        "Pol_pos": "-gene_len", "Pol_on":-1, "Pol_off":1,
                        "U1_1":"-U1_1", "U2_1": "-U2_1","U1_2":"-U1_2", "U2_2": "-U2_2",
                        "U11p":"U1_1", "U21p": "U2_1","U12p":"U1_2", "U22p": "U2_2"},
                       name = "Termination: full retention")
         
        #Posttranscriptional reactions        
        s.add_reaction("(ret+ret_i1-U11p)*u1_1_br", {"U11p": 1}, "U11p binding")
        s.add_reaction("U11p*u1ur", {"U11p": -1}, "U11p unbinding")
        s.add_reaction("(ret+ret_i1-U21p)*u2_1_br", {"U21p": 1}, "U21p binding")
        s.add_reaction("U21p*u2ur", {"U21p": -1}, "U21p unbinding")
        s.add_reaction("(ret+ret_i2-U12p)*u1_2_br", {"U12p": 1}, "U12p binding")
        s.add_reaction("U12p*u1ur", {"U12p": -1}, "U12p unbinding")
        s.add_reaction("(ret+ret_i2-U22p)*u2_2_br", {"U22p": 1}, "U22p binding")
        s.add_reaction("U22p*u2ur", {"U22p": -1}, "U22p unbinding")
        
        s.add_reaction("spl_rate*U11p * U21p*ret/((ret+ret_i1)**2)",
                       {"ret": -1, "ret_i2": 1, "U11p":-1, "U21p":-1}, "PostTr. ret -> ret_i2")
        s.add_reaction("spl_rate*U11p * U21p*ret_i1/((ret+ret_i1)**2)",
                       {"ret_i1": -1, "Incl": 1, "U11p":-1, "U21p":-1}, "PostTr. ret_i1 -> Incl")
        s.add_reaction("spl_rate*U12p * U22p*ret/((ret+ret_i2)**2)",
                       {"ret": -1, "ret_i1": 1, "U12p":-1, "U22p":-1}, "PostTr. ret -> ret_i1")
        s.add_reaction("spl_rate*U12p * U22p*ret_i2/((ret+ret_i2)**2)",
                       {"ret_i2": -1, "Incl": 1, "U12p":-1, "U22p":-1}, "PostTr. ret_i2 -> Incl")
        
        s.add_reaction("spl_rate * U11p*ret/(ret+ret_i1) * U22p/(ret+ret_i2)",
                       {"ret": -1, "Skip": 1, "U11p":-1, "U22p":-1,
                        "U21p":"-round(U21p/(ret+ret_i1) if U21p > 0 else 0)",
                        "U12p":"-round(U12p/(ret+ret_i2) if U12p > 0 else 0)"},
                       "PostTr. ret -> Skip")
#        s.add_reaction("((u1_1_br+u2_2_br)/2  + 1/(spl_rate)) * ret", {"ret": -1, "Skip": 1}, "PostTr. ret -> Skip")
#        s.add_reaction("((u1_1_br+u2_1_br)/2  + 1/(spl_rate)) * ret", {"ret": -1, "ret_i2": 1}, "PostTr. ret -> ret_i2")
#        s.add_reaction("((u1_2_br+u2_2_br)/2  + 1/(spl_rate)) * ret", {"ret": -1, "ret_i1": 1}, "PostTr. ret -> ret_i1")
#        s.add_reaction("((u1_2_br+u2_2_br)/2  + 1/(spl_rate)) * ret_i2", {"ret_i2": -1, "Incl": 1}, "PostTr. ret_i2 -> Incl")
#        s.add_reaction("((u1_1_br+u2_1_br)/2  + 1/(spl_rate)) * ret_i1", {"ret_i1": -1, "Incl": 1}, "PostTr. ret_i1 -> Incl")
        #Degradation
        s.add_reaction("d1 * Incl", {"Incl": -1}, "Incl degr.")
        s.add_reaction("d2 * Skip", {"Skip": -1}, "Skip degr.")
#        s.add_reaction("s3 * mRNA", {"mRNA": -1, "ret": 1})
#        s.add_reaction("d3 * ret", {"ret": -1}, "ret degr.")
#        s.add_reaction("d3 * ret_i1", {"ret_i1": -1}, "ret_i1 degr.")
#        s.add_reaction("d3 * ret_i2", {"ret_i2": -1}, "ret_i2 degr")
#        
        
    elif(name == "CoTrSplicing_2"):
        s = get_exmpl_sim("CoTrSplicing")
        s.set_param("u2_pol_br", 1) #binding rate of U2 + Pol
        s.set_param("u2_pol_ur", 0.01)
        s.set_param("u2pol_br", 1) #max binding rate of U2Pol + mRNA
        s.set_param("u2_pol_d", 20) # optimal distance from Pol2
        s.set_param("u2_pol_d_r", 10)
        s.add_reaction("Pol_on * u2_pol_br", {"U2_Pol":[1, None]}, "Pol + U2")
        s.add_reaction("U2_Pol * u2_pol_ur", {"U2_Pol":-1}, "Pol/U2 diss.")
        s.add_reaction("U2_Pol * u2pol_br * norm_proximity(Pol_pos-u2_pol_d, u2_1_bs_pos, u2_pol_d_r, 3)",
                        {"U2_1":[1,None], "U2_Pol":-1,
                         "Pol_pos":["u2_1_bs_pos", "-u2_1_bs_pos"],
                         "Intr1":[1,-1]},
                        "U2onPol to U2_1")
        s.add_reaction("U2_Pol * u2pol_br * norm_proximity(Pol_pos-u2_pol_d, u2_2_bs_pos, u2_pol_d_r, 3) ",
                        {"U2_2":[1,None], "U2_Pol":-1,
                         "Pol_pos":["u2_2_bs_pos", "-u2_2_bs_pos"],
                         "Intr2":[1,-1]},
                        "U2onPol to U2_2")
        
#        s.add_reaction()
        
    elif(name == "CoTrSplicing_3"):
        #TODO
        # working in progress ....................
        s = get_exmpl_sim("CoTrSplicing_2")
        s.set_param("rbp_pos", 1500)
        s.set_param("rbp_radius", 100)
        s.set_param("rbp_br", 1)
        s.set_param("rbp_ur", 0.2)
        
        
    return s