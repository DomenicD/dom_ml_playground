class Normalizer(object):
    """Used to normailze the data going into and coming out of a neural
    network.

    If out_range is not specified, it will use the in_range.
    If out_offset is not specified, it will use the in_offset.

    """
    def __init__(self, in_lower = 0, in_upper = 1,
                 out_lower = 0, out_upper = 1,
                 norm_lower = -3, norm_upper = 3):
        
        self.in_offset = in_lower
        self.in_range = in_upper - in_lower

        self.out_offset = out_lower
        self.out_range = out_upper - out_lower

        self.norm_offset = norm_lower
        self.norm_range = norm_upper - norm_lower
        
    
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