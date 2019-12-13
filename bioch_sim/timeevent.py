import re

class TimeEvent(object):
    
    _count = 0    
    def __init__(self, t, a, name = None):
        self._id = TimeEvent._count
        if (name is None):
            name = "TimeEvent" + str(TimeEvent._count)
            TimeEvent._count+=1
        self.name = name
        self.consts = None
        self.set_time(t)
        self.action = a
        
    def __lt__(self, te):
        return self.t < te.t
    def __le__(self, te):
        return self.t <= te.t
    def __gt__(self, te):
        return self.t > te.t
    def __ge__(self, te):
        return self.t >= te.t
    def __str__(self):
        return self.name + ":" + " at " + str(self.t) + "\nAction: " + self.action 
    def __repr__(self):
        return  self.__str__()
    def set_time(self, t):
        self.t = t
    def set_action(self, a):
        self.action = a
    def set_constants(self, consts):
        self.consts = consts
        if(self.__t_str is not None):
            self.__update_expr()
    def __update_expr(self):
        s = self.__t_str
        for  k, v in self.consts.items():
            s = re.sub("\\b" + k + "\\b", "%s[\"%s\"]" % ("self.consts",k), s)
        self.__expr_str = s
        
    def __get_t(self):
        if(self.__t_str is not None):
#            print(self.__expr_str)
            return eval(self.__expr_str)
        else:
            return self.__t
    def __set_t(self, t):
        if(type(t) is str):
            self.__t_str = t
            if(self.consts is not None):
                self.__update_expr()
        else:
            self.__t = t
            self.__t_str = None
    t = property(__get_t, __set_t)
