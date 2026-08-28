class Domain:
    def __init__(self, domain_id: int, n: int, x_interval: tuple, y_interval: tuple = (-float("inf"), float("inf"))):
        self.domain_id = domain_id
        self.n = n 
        self.x_interval = x_interval
        self.y_interval = y_interval
