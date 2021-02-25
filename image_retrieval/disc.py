class disc:
    def __init__(self):
        self.fname = 'disc.dat'
        with open(self.fname) as f:
            self.lines = f.readlines()
        self.lines = [line.rstrip('\n') for line in self.lines]
        self.lines = list(map(int, self.lines))
        self.min = self.lines[0]
        self.max = self.lines[1]
        self.num = self.lines[2]
        self.res = self.lines[3]
        self.dat = self.lines[4:]

    def getdisc(self, ratio):
        if ratio <= self.min:
            return 0
        elif ratio >= self.max:
            return self.num-1
        else:
            return self.dat[int(((ratio-self.min)/(self.max-self.min))*self.res)]
