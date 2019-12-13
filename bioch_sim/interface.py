from tkinter import ttk

import tkinter as tk
from tkinter.colorchooser import askcolor
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.colors as colors
import pylab as plt

class SimInterface(tk.Frame):
    def __init__(self, sim):
        super().__init__()
        self.sim = sim
        self.initUI(sim)
    
    def initUI(self, sim):
        self.sim = sim
        self.master.title(sim.name)
        self.pack(fill=tk.BOTH, expand=True)
                
        self.columnconfigure(1, weight=1)
        self.columnconfigure(3, pad=7)
        self.rowconfigure(3, weight=1)
        self.rowconfigure(5, pad=7)
        
        f_settings = self._create_settings_f(self, sim)
        f_settings.pack(side=tk.TOP, fill=tk.X)
        f_places = self._create_graph_selection(self, sim)
        f_places.pack(fill=tk.X, side=tk.LEFT)
        
        f_plot = tk.Frame(self)
        f_plot.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)
        par_sbar = tk.Scrollbar(self, orient="vertical" )
        scroll_geometry = (0, 0, 1000, 1000)
        f_params_container = tk.Canvas(self, scrollregion=scroll_geometry,
                                       yscrollcommand=par_sbar.set)
        
        f_params = tk.Frame(f_params_container)
        f_params.pack(side=tk.BOTTOM, fill = tk.X, expand=True)
        
        par_sbar.pack(side=tk.RIGHT, fill=tk.Y)
        f_params_container.configure(yscrollcommand=par_sbar.set)
        f_params_container.pack(side = tk.TOP, fill = tk.BOTH, expand = True)
        par_sbar.config(command=f_params_container.yview)
        
#        def key_pressed(e):
#            print("key pressed: ", e.char)
#            if(e.char == "\n"): self.update(True)
#        f_params.bind('<Return>', lambda e: key_pressed(e) ) 
        
        self._show_sp = [[],[]]
        self._show_sp
        i = 0
        self.par_entries = {}
       
        for k, v in sim.params.items():
#            row = tk.Frame(f_params)
#            row.pack(side=tk.TOP, fill=tk.X, padx = 1, pady=1)
            label = tk.Label(f_params, text=k)
            label.grid(row=i, column=0)
            entr = tk.Entry(f_params)
            entr.insert(0,v)
            entr.grid(row=i, column=1)
            entr.bind('<Return>', lambda e: self.update(True) ) 
            self.par_entries[k]= entr
            i +=1
            
       
            
        fig = plt.Figure(figsize=(5,5))
        self._fig = fig
        canvas = FigureCanvasTkAgg(fig, f_plot)
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self._canvas = canvas
        self._plot_sim()
        
        f_control = tk.Frame(self)
        f_control.pack(side =tk.BOTTOM, fill =tk.X)
        b_update = tk.Button(f_control, text="Update",
                             command = lambda: self.update(True))
        b_update.pack(side=tk.RIGHT)
    def _create_settings_f(self, master,sim):
        self._setting_e ={}
        #common settings frame
        f_settings = tk.Frame(self)
        l_runtime  = tk.Label(f_settings,text = "runtime:" )
        l_runtime.grid(row=0, column=0)
        l_raster = tk.Label(f_settings, text = "raster:")
        l_raster.grid(row=1, column=0)
        
        e_runtime = tk.Entry(f_settings)
        e_runtime.insert(0,str(sim.runtime))
        e_runtime.bind('<Return>', lambda e: self.update(True) ) 
        e_runtime.grid(row=0, column=1)
        self._setting_e["runtime"] = e_runtime
        
        e_raster = tk.Entry(f_settings)
        e_raster.insert(0,str(sim.raster_count))
        e_raster.bind('<Return>', lambda e: self.update(True)) 
        e_raster.grid(row=1, column=1)
        self._setting_e["raster_count"] = e_raster
        return f_settings
    
    def _create_graph_selection(self, master, sim):
        self._spezies_checkb = {}
        self._spezies_checkb2 = {}
        self._spezies_col_b = {}
        self.pl_entries = {}
        frame = tk.Frame(master)
        for i, (k, v) in enumerate(sim.init_state.items()):
            c = sim._get_color(k)
            checked = tk.IntVar()
            check_box = tk.Checkbutton(frame, text="", variable = checked, #command = self.update)
                                       command= lambda : self.update(False))
#            check_box.select()
            check_box.grid(row=i, column=0)
            self._spezies_checkb[k] = checked
            checked = tk.IntVar()
            check_box = tk.Checkbutton(frame, text="", variable = checked, #command = self.update)
                                       command= lambda : self.update(False))
#            check_box.select()
            check_box.grid(row=i, column=1)
            self._spezies_checkb2[k] = checked
            label = tk.Label(frame, text=k)
            label.grid(row=i, column =2)
            entr = tk.Entry(frame, width=4)
            entr.insert(0,v)
            entr.grid(row=i, column=3)
            self.pl_entries[k]= entr
            b_col = tk.Button(frame, height =1, width=1, bg = colors.to_hex(c),
                              command = lambda key=k: self._new_color(key))
            b_col.grid(row=i, column=3)
            self._spezies_col_b[k] = b_col
            
        #create transition(rate) selection
        i+=1
        self._transtion_checks = []
        checked = tk.IntVar()
        check_box = tk.Checkbutton(frame, text="", variable = checked, #command = self.update)
                                   command= lambda : self.update(False))
        check_box.grid(row=i, column=0)
        self._transtion_checks.append(checked)
        checked = tk.IntVar()
        check_box = tk.Checkbutton(frame, text="", variable = checked, #command = self.update)
                                   command= lambda : self.update(False))
        check_box.grid(row=i, column=1)
        self._transtion_checks.append(checked)
        
        transition_cb = ttk.Combobox(frame, values=list(sim._transitions))
        transition_cb.current(0)
        transition_cb.bind("<<ComboboxSelected>>", lambda e: self.update(False))
        transition_cb.grid(row=i, column=2)
        self._tr_cb = transition_cb
        
        #free expression Entry
        i+=1
        self._expr_checks = []
        checked = tk.IntVar()
        check_box = tk.Checkbutton(frame, text="", variable = checked, #command = self.update)
                                   command= lambda : self.update(False))
        check_box.grid(row=i, column=0)
        self._expr_checks.append(checked)
        checked = tk.IntVar()
        check_box = tk.Checkbutton(frame, text="", variable = checked, #command = self.update)
                                   command= lambda : self.update(False))
        check_box.grid(row=i, column=1)
        self._expr_checks.append(checked)
        
        expr_e = tk.Entry(frame, width=20)
        expr_e.bind("<Return>", lambda e: self.update(False))
        expr_e.grid(row=i, column=2)
        self._expr_e = expr_e
            
        return frame
            
    def _new_color(self, name):
        b_c = self._spezies_col_b[name]
        col_new = askcolor(b_c.cget("bg"))[1]
        if col_new is not None:
            b_c.configure(bg=col_new)
            self.sim._set_color(name, col_new)
        self.update(False)
    
    def _plot_sim(self):
        self._fig.clear()
        ax = self._fig.add_subplot(111)
#        ax.clear()
        print(self._show_sp)
        self.sim.plot_course(ax = ax, products = self._show_sp[0], products2=self._show_sp[1])
        ax.legend([])
        ax.set_ylabel("#", rotation = 0)
        self._canvas.draw()
    
    def update(self, sim=False):
        self.fetch_places()
        if(sim):
            self.fetch_pars()
            self.fetch_settings()
            self.sim.simulate()
        print("update..")
        self._plot_sim()
    def fetch_settings(self):
        sim = self.sim
        for k, v in self._setting_e.items():
            sim.__dict__[k] = eval(v.get())
        sim.set_raster()
        
    def fetch_places(self):
        sim = self.sim
        self._show_sp = []
        show_sp = []
        for k, checked in self._spezies_checkb.items():
            if checked.get():
                show_sp.append(k)
                
        show_sp2 = []
        for k, checked in self._spezies_checkb2.items():
            if checked.get():
                show_sp2.append(k)
                
        self._show_sp=[show_sp, show_sp2]
        
        for i, tr_cb in enumerate(self._transtion_checks):
            if(tr_cb.get()):
                self._show_sp[i].append(sim._transitions[self._tr_cb.get()]["rate_ex"])
        for i, expr_cb in enumerate(self._expr_checks):
            if(expr_cb.get()):
                self._show_sp[i].append(self._expr_e.get())
        
    def fetch_pars(self):
        sim = self.sim
        for k, e in self.par_entries.items():
            v = eval(e.get())
            sim.set_param(k,v)
        
            
        
