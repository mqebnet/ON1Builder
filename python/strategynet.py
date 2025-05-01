# strategynet.py
import asyncio
import time
import numpy as np
import random
from typing import Any, Dict, List, Callable
from decimal import Decimal

from apiconfig import APIConfig
from transactioncore import TransactionCore
from safetynet import SafetyNet
from marketmonitor import MarketMonitor
from loggingconfig import setup_logging

logger = setup_logging("Strategy_Net", level="DEBUG")  

class StrategyPerformanceMetrics:
    """Tracks performance metrics for a strategy."""
    def __init__(self) -> None:
        self.successes: int = 0
        self.failures: int = 0
        self.profit: Decimal = Decimal("0")
        self.avg_execution_time: float = 0.0
        self.success_rate: float = 0.0
        self.total_executions: int = 0

class StrategyConfiguration:
    """Holds configurable parameters for strategy selection."""
    def __init__(self) -> None:
        self.decay_factor: float = 0.95
        self.learning_rate: float = 0.01
        self.exploration_rate: float = 0.1
        self.FRONT_RUN_OPPORTUNITY_SCORE_THRESHOLD: int = 75
        self.VOLATILITY_FRONT_RUN_SCORE_THRESHOLD: int = 75
        self.AGGRESSIVE_FRONT_RUN_RISK_SCORE_THRESHOLD: float = 0.7

class StrategyNet:
    """
    Orchestrates strategy selection and execution via reinforcement learning.
    """
    def __init__(
        self,
        transactioncore: TransactionCore,
        marketmonitor: MarketMonitor,
        safetynet: SafetyNet,
        apiconfig: APIConfig
    ) -> None:
        self.transactioncore = transactioncore
        self.marketmonitor = marketmonitor
        self.safetynet = safetynet
        self.apiconfig = apiconfig
        self.strategy_types: List[str] = ["eth_transaction", "front_run", "back_run", "sandwich_attack"]
        self._strategy_registry: Dict[str, List[Callable[[Dict[str, Any]], asyncio.Future]]] = {
            "eth_transaction": [self.transactioncore.handle_eth_transaction],
            "front_run": [
                self.transactioncore.front_run,
                self.transactioncore.aggressive_front_run,
                self.transactioncore.predictive_front_run,
                self.transactioncore.volatility_front_run
            ],
            "back_run": [
                self.transactioncore.back_run,
                self.transactioncore.price_dip_back_run,
                self.transactioncore.flashloan_back_run,
                self.transactioncore.high_volume_back_run
            ],
            "sandwich_attack": [
                self.transactioncore.execute_sandwich_attack
            ]
        }
        self.strategy_performance: Dict[str, StrategyPerformanceMetrics] = {
            stype: StrategyPerformanceMetrics() for stype in self.strategy_types
        }
        self.reinforcement_weights: Dict[str, np.ndarray] = {
            stype: np.ones(len(self._strategy_registry[stype]))
            for stype in self.strategy_types
        }
        self.configuration: StrategyConfiguration = StrategyConfiguration()
        logger.debug("StrategyNet initialized.")

    async def initialize(self) -> None:
        """Performs any asynchronous initialization required."""
        logger.info("StrategyNet initialization complete.")

    def get_strategies(self, strategy_type: str) -> List[Callable[[Dict[str, Any]], asyncio.Future]]:
        """Returns available strategies for a given type."""
        return self._strategy_registry.get(strategy_type, [])

    async def _select_best_strategy(
        self,
        strategies: List[Callable[[Dict[str, Any]], asyncio.Future]],
        strategy_type: str
    ) -> Callable[[Dict[str, Any]], asyncio.Future]:
        """Selects a strategy based on reinforcement learning probabilities."""
        weights = self.reinforcement_weights[strategy_type]
        if random.random() < self.configuration.exploration_rate:
            logger.debug("Exploration: randomly selecting a strategy.")
            return random.choice(strategies)
        max_weight = np.max(weights)
        exp_weights = np.exp(weights - max_weight)
        probabilities = exp_weights / exp_weights.sum()
        index = np.random.choice(len(strategies), p=probabilities)
        selected = strategies[index]
        logger.debug(f"Selected strategy '{selected.__name__}' with weight {weights[index]:.4f}.")
        return selected

    def _calculate_reward(self, success: bool, profit: Decimal, execution_time: float) -> float:
        """Calculates reward for strategy update."""
        base_reward = float(profit) if success else -0.1
        time_penalty = 0.01 * execution_time
        risk_penalty = 0.05
        worst_case_penalty = 0.05 * max(0.0, execution_time - 2.0)
        total = base_reward - time_penalty - risk_penalty - worst_case_penalty
        logger.debug(f"Reward computed: {total:.4f}")
        return total

    async def _update_strategy_metrics(
        self,
        strategy_name: str,
        strategy_type: str,
        success: bool,
        profit: Decimal,
        execution_time: float
    ) -> None:
        """Updates performance metrics and reinforcement weights."""
        metrics = self.strategy_performance[strategy_type]
        metrics.total_executions += 1
        if success:
            metrics.successes += 1
            metrics.profit += profit
        else:
            metrics.failures += 1
        df = self.configuration.decay_factor
        metrics.avg_execution_time = metrics.avg_execution_time * df + execution_time * (1 - df)
        metrics.success_rate = metrics.successes / metrics.total_executions
        idx = self.get_strategy_index(strategy_name, strategy_type)
        if idx >= 0:
            current = self.reinforcement_weights[strategy_type][idx]
            reward = self._calculate_reward(success, profit, execution_time)
            gamma = 0.9
            updated = current + self.configuration.learning_rate * (reward + gamma * current - current)
            self.reinforcement_weights[strategy_type][idx] = max(0.1, updated)
            logger.debug(f"Updated weight for {strategy_name} from {current:.4f} to {updated:.4f}.")

    def get_strategy_index(self, strategy_name: str, strategy_type: str) -> int:
        """Finds the index of a strategy in the registry."""
        for idx, strat in enumerate(self.get_strategies(strategy_type)):
            if strat.__name__ == strategy_name:
                return idx
        logger.warning(f"Strategy {strategy_name} not found in type {strategy_type}.")
        return -1

    async def execute_best_strategy(self, target_tx: Dict[str, Any], strategy_type: str) -> bool:
        """Executes the best strategy for a profitable transaction."""
        strategies = self.get_strategies(strategy_type)
        if not strategies:
            logger.debug(f"No strategies for type {strategy_type}.")
            return False
        start = time.time()
        strategy = await self._select_best_strategy(strategies, strategy_type)
        before = Decimal(self.transactioncore.current_profit)
        success = await strategy(target_tx)
        after = Decimal(self.transactioncore.current_profit)
        exec_time = time.time() - start
        profit = after - before
        await self._update_strategy_metrics(strategy.__name__, strategy_type, success, profit, exec_time)
        return success
