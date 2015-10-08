class Normalizer(object):
    """Used to normailze the data going into and coming out of a neural
    network.

    If out_range is not specified, it will use the in_range.
    If out_offset is not specified, it will use the in_offset.

    """
    def __init__(self, in_range = 1, in_offset = 0,
                 out_range = None, out_offset = None):
        self.in_range = in_range
        self.in_offset = in_offset
        
        if out_range == None:
            self.out_range = in_range
        else:
            self.out_range = out_range
        
        if out_offset == None:
            self.out_offset = in_offset
        else:
            self.out_offset = out_offset

        
    def norm_input(self, x):
       return (x + self.in_offset) / self.in_range
   

    def norm_output(self, x):
       return (x * self.out_range) - self.out_offset


    def denorm_input(self, x):
       return (x * self.in_range) - self.in_offset
   

    def denorm_output(self, x):
       return (x + self.out_offset) / self.out_range