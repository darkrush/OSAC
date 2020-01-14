import numpy

class NEReplay:  # stored as ( s, a, r, s_ ) in SumTree
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = numpy.zeros(capacity, dtype=object)
        self.write = 0
        self.number = 0
    def update(self, idx, error):
        pass
    def add(self,error, sample):
        self.data[self.write] = sample
        self.number = min(self.number+1,self.capacity)
        if self.number == self.capacity:
            self.write = numpy.random.randint(self.capacity)
        else:
            self.write = self.write+1
        
    def sample(self, n):
        batch = []
        idxs = numpy.random.randint(0,self.number,[n])
        is_weight = numpy.ones_like(idxs)
        for idx in range(n):
            batch.append(self.data[idxs[idx]])
        return batch, idxs, is_weight