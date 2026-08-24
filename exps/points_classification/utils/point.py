class Point:
    def __init__(self, x1, x2, domain, label) -> None:
        # domain : source (0) or target (1)
        # label (=y) : 0 or 1
        self.x1 = x1
        self.x2 = x2 
        self.domain = domain 
        self.label = label
