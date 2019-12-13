#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 12:11:08 2019

@author: timur
"""

import matplotlib.colors as colors
import re
import os
import snakes
import snakes.plugins
snakes.plugins.load("gv","snakes.nets","nets")
#    snakes.plugins.load('clusters', 'nets', 'snk')
import nets as pns

from .base import SimParam

class SimParam(SimParam):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def draw_pn(self, filename=None, rates=False, rotation = False,
                engine=('neato', 'dot', 'circo', 'twopi', 'fdp'),
                draw_neutral_arcs = True,
                draw_inhibition_arcs = True,
                draw_flush_arcs = True,
                **kwargs):
        if type(engine) is str:
            engine = (engine,)
        self.compile_system()
        #https://www.ibisc.univ-evry.fr/~fpommereau/SNAKES/API/plugins/gv.html
        if filename is None:
            filename = self.name + ".png"
            filename = os.path.join("pn_images", filename)
            path = os.path.dirname(filename)
            if(not os.path.exists(path)):
                os.makedirs(path)
            
        pn = pns.PetriNet(self.name)
        for i, (p, v) in enumerate(self.init_state.items()):
            cluster = self._clusters[p] if p in self._clusters else ()
#            pn.add_place(pns.Place(p,v), cluster=cluster)
            pn.add_place(pns.Place(p,v), cluster=cluster)
        
        for i, (tr_name, pre, post) in enumerate(zip(self._transitions.keys(), self._pre, self._post)):
            name = tr_name
            tr = self._transitions[tr_name]
            cluster = self._clusters[name] if name in self._clusters else ()
            if(rates):
                pn.add_transition(pns.Transition(name, pns.Expression(tr["rate"])),
                                                 cluster = cluster)
            else:
                pn.add_transition(pns.Transition(name), cluster=cluster)
                
            #creating arcs
            for pr, pst, subs in zip(pre, post, list(self.init_state)):
                if(pr == pst and pr != 0): #neutral 
                    if(draw_neutral_arcs):
                        v = pns.Value(pr)
                        v._role = "neutral"
                        pn.add_input(subs, name, v)
                else:
                    # ugly stuff starts :(
                    if(pr == "2*" + subs ): #inhibition
                        if(draw_inhibition_arcs):
                            v = pns.Value("<inhibits>")
                            v._role = "inhibition"
                            pn.add_input(subs, name, v)
                    elif(pr == subs): #flush
                        if(draw_flush_arcs):
                            flush = pns.Flush("0")
                            flush._role = "flush"
                            pn.add_output(subs, name, flush)
                    elif(pr != 0):
                        pn.add_input(subs, name, pns.Value(pr))
                    if(pst != 0):
                        pn.add_output(subs, name, pns.Value(pst))
                    
         
        #documentation of attr                
        #http://www.graphviz.org/doc/info/attrs.html        
        def draw_place (place, attr) :
#            print(attr)
            attr['label'] = place.name
            attr['color'] = colors.to_hex(self._get_color(place.name))
        def draw_transition (trans, attr) :
#            print(attr)
            if str(trans.guard) == 'True' :
                attr['label'] = trans.name
            else :
                attr['label'] = '%s\n%s' % (trans.name, trans.guard)
        def draw_arc(arc, attr):
            #arrow styles
            #http://www.graphviz.org/doc/info/attrs.html#k:arrowType
            if(hasattr(arc, "_role")):
                if(arc._role == "inhibition"):
                    attr["arrowhead"] = "tee"
#                    attr["arrowhead"] = "odot"
                    attr["style"] = "bold"
                    attr["label"] = "inhibition"
                    pass
                if(arc._role == "flush"):
#                    attr["arrowhead"] = "odot"
                    attr["label"] = "0"
                    attr["arrowhead"] = "empty"
                    attr["style"] = "dashed"
                    pass
                if(arc._role == "neutral"):
                    attr["dir"] = "both"
#                    attr["arrowhead"] = "box"
#                    attr["arrowtail"] = "box"
                    attr["style"] = "dotted"
                    attr["arrowhead"] = "ediamond"
                    attr["arrowtail"] = "ediamond"
                    pass
        def draw_graph(g, attr):
#            print("AAAAAAAAA", g)
#            print(attr)
            attr["rotate"] = 0
            attr["style"] = "invis"
#            attr["style"] = "dashed"
            attr["ratio"] = 2
            attr["rankdir"] = "LR"
#            attr["bgcolor"] = "#ff0000"
#            attr["label"] = "KOMM SCHON"
        if(rotation):
            pn.transpose()
        for e in engine:
            f_name = re.sub("(\.\w+)$", "_"+ e + "\\1", filename)
            pn.draw(f_name, engine = e, debug=False,
                place_attr=draw_place,
                trans_attr=draw_transition ,
                arc_attr = draw_arc,
                graph_attr = draw_graph,
                cluster_attr = draw_graph,
                **kwargs)
        return pn