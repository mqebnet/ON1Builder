# File: python/strategynet.py

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
import logging

logger = setup_logging("StrategyNet", level=logging.DEBUG)

class StrategyPerformanceMetrics:
    def __init__(self):
        self.successes: int = 0
        self.failures: int = 0
        self.profit: Decimal = Decimal("0")
        self.avg_execution_time: float = 0.0
        self.success_rate: float = 0.0
        self.total_executions: int = 0

class StrategyConfiguration:
    def __init__(self):
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
    def __init__(self,
                 transactioncore: TransactionCore,
                 marketmonitor: MarketMonitor,
                 safetynet: SafetyNet,
                 apiconfig: APIConfig) -> None:
        self.transactioncore = transactioncore
        self.marketmonitor = marketmonitor
        self.safetynet = safetynet
        self.apiconfig = apiconfig
        self.strategy_types = ["eth_transaction", "front_run", "back_run", "sandwich_attack"]
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
        logger.info("StrategyNet initialization complete.")

    def get_strategies(self, strategy_type: str) -> List[Callable[[Dict[str, Any]], asyncio.Future]]:
        return self._strategy_registry.get(strategy_type, [])

    async def _select_best_strategy(self, strategies: List[Callable[[Dict[str, Any]], asyncio.Future]], strategy_type: str) -> Callable[[Dict[str, Any]], asyncio.Future]:
        weights = self.reinforcement_weights[strategy_type]
        if random.random() < self.configuration.exploration_rate:
            logger.debug("Exploration: randomly selecting a strategy.")
            return random.choice(strategies)
        max_weight = np.max(weights)
        exp_weights = np.exp(weights - max_weight)
        probabilities = exp_weights / exp_weights.sum()
        selected_index = np.random.choice(len(strategies), p=probabilities)
        selected_strategy = strategies[selected_index]
        logger.debug(f"Selected strategy '{selected_strategy.__name__}' (weight: {weights[selected_index]:.4f}).")
        return selected_strategy

    def _calculate_reward(self, success: bool, profit: Decimal, execution_time: float) -> float:
        base_reward = float(profit) if success else -0.1
        time_penalty = 0.01 * execution_time
        risk_penalty = 0.05
        worst_case_threshold = 2.0
        worst_case_penalty = 0.05 * max(0.0, execution_time - worst_case_threshold)
        total_reward = base_reward - time_penalty - risk_penalty - worst_case_penalty
        logger.debug(f"Reward computed: base: {base_reward:.4f}, time_penalty: {time_penalty:.4f}, "
                     f"risk_penalty: {risk_penalty:.4f}, worst_case_penalty: {worst_case_penalty:.4f}, "
                     f"total: {total_reward:.4f}")
        return total_reward

    async def _update_strategy_metrics(self,
                                       strategy_name: str,
                                       strategy_type: str,
                                       success: bool,
                                       profit: Decimal,
                                       execution_time: float) -> None:
        metrics = self.strategy_performance[strategy_type]
        metrics.total_executions += 1
        if success:
            metrics.successes += 1
            metrics.profit += profit
        else:
            metrics.failures += 1

        decay = self.configuration.decay_factor
        metrics.avg_execution_time = (metrics.avg_execution_time * decay +
                                      execution_time * (1 - decay))
        metrics.success_rate = metrics.successes / metrics.total_executions

        strategy_index = self.get_strategy_index(strategy_name, strategy_type)
        if strategy_index >= 0:
            current_weight = self.reinforcement_weights[strategy_type][strategy_index]
            reward = self._calculate_reward(success, profit, execution_time)
            gamma = 0.9
            next_max_q = current_weight
            learning_rate = self.configuration.learning_rate
            updated_weight = current_weight + learning_rate * (reward + gamma * next_max_q - current_weight)
            self.reinforcement_weights[strategy_type][strategy_index] = max(0.1, updated_weight)
            logger.debug(f"Updated weight for {strategy_name} (type: {strategy_type}, index: {strategy_index}) "
                         f"from {current_weight:.4f} to {updated_weight:.4f}")
        logger.debug(f"Strategy metrics for {strategy_type}: total_executions: {metrics.total_executions}, "
                     f"success_rate: {metrics.success_rate:.4f}, avg_execution_time: {metrics.avg_execution_time:.4f}")

    def get_strategy_index(self, strategy_name: str, strategy_type: str) -> int:
        strategies = self.get_strategies(strategy_type)
        for idx, strat in enumerate(strategies):
            if strat.__name__ == strategy_name:
                return idx
        logger.warning(f"Strategy {strategy_name} not found for type {strategy_type}")
        return -1

    async def execute_best_strategy(self, target_tx: Dict[str, Any], strategy_type: str) -> bool:
        strategies = self.get_strategies(strategy_type)
        if not strategies:
            logger.debug(f"No strategies available for type {strategy_type}.")
            return False
        start_time = time.time()
        selected_strategy = await self._select_best_strategy(strategies, strategy_type)
        profit_before = Decimal(self.transactioncore.current_profit)
        success = await selected_strategy(target_tx)
        profit_after = Decimal(self.transactioncore.current_profit)
        execution_time = time.time() - start_time
        profit_made = profit_after - profit_before
        await self._update_strategy_metrics(selected_strategy.__name__, strategy_type, success, profit_made, execution_time)
        return success
