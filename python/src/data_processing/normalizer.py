class Normalizer(object):
    """Used to normailze the data going into and coming out of a neural
    network.  

    """
    def __init__(self, in_min = 0, in_max = 1,
                 out_min = 0, out_max = 1,
                 norm_min = 0, norm_max = 1):
        
        self.in_offset = in_min
        self.in_range = in_max - in_min

        self.out_offset = out_min
        self.out_range = out_max - out_min

        self.norm_offset = norm_min
        self.norm_range = norm_max - norm_min
        
    
    def norm_input(self, x):
        return self._normalize((x - self.in_offset) / float(self.in_range))
   

    def norm_output(self, x):
        return (x * self.out_range) + self.out_offset


    def denorm_input(self, x):
        x = self._denormalize(x)
        return (x * self.in_range) + self.in_offset
   

    def denorm_output(self, x):
        return (x - self.out_offset) / float(self.out_range)


    def _normalize(self, x):
        return (x * self.norm_range) + self.norm_offset


    def _denormalize(self, x):
        return (x - self.norm_offset) / float(self.norm_range)