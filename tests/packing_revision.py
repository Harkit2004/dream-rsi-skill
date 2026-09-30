from dream_rsi.policy import BudgetAwarePolicy


class OptimalPolicy(BudgetAwarePolicy):
    def __init__(self, config=None):
        super().__init__({"beta": 1.0})
