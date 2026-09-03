class Logger:
    def __init__(self, verbose: bool) -> None:
        self.verbose = verbose

    @staticmethod
    def static_log(verbose: bool, *values: object, sep: str = " ") -> None:
        if verbose:
            print(*values, sep=sep)

    def log(self, *values: object, sep: str = " ") -> None:
        Logger.static_log(self.verbose, *values, sep=sep)

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.log(*args, **kwargs)

