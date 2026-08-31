from exps.points_classification.utils.domain import Domain


class MoonDomain(Domain):
    def __init__(self, domain_id: int, n: int, tilt_rad: float, x_interval: tuple = (-float("inf"), float("inf")), y_interval: tuple = ...):
        super().__init__(domain_id, n, x_interval, y_interval)
        self.tilt_rad = tilt_rad
