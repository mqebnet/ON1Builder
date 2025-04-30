import asyncio
import time
import numpy as np
import random
from typing import Any, Dict, List, Callable, Coroutine, Optional, Tuple
from decimal import Decimal
from dataclasses import dataclass, field
import os
from pathlib import Path

# Assuming configuration is now imported via facade
from configuration import Configuration
from transactioncore import TransactionCore
from safetynet import SafetyNet
from marketmonitor import MarketMonitor
from loggingconfig import setup_logging
import logging

logger = setup_logging("StrategyNet", level=logging.DEBUG)

# A1: Replace simple classes with dataclasses
@dataclass(frozen=True, slots=True)
class StrategyPerformanceMetrics:
    """Holds performance metrics for a strategy type."""
    successes: int = 0
    failures: int = 0
    total_profit: Decimal = Decimal("0") # Renamed from profit for clarity
    avg_execution_time: float = 0.0
    success_rate: float = 0.0
    total_executions: int = 0

@dataclass(frozen=True, slots=True)
class StrategyConfiguration:
    """Holds configuration parameters for StrategyNet."""
    decay_factor: float = 0.95
    learning_rate: float = 0.01
    exploration_rate: float = 0.1
    # These seem specific to certain strategies, potentially move later
    FRONT_RUN_OPPORTUNITY_SCORE_THRESHOLD: int = 75
    VOLATILITY_FRONT_RUN_SCORE_THRESHOLD: int = 75
    AGGRESSIVE_FRONT_RUN_RISK_SCORE_THRESHOLD: float = 0.7


class StrategyNet:
    """
    Orchestrates strategy selection and execution via reinforcement learning.
    """
    WEIGHTS_FILENAME = "weights.npy"
    SAVE_INTERVAL_SECONDS = 60

    def __init__(self,
                 transactioncore: TransactionCore,
                 marketmonitor: MarketMonitor,
                 safetynet: SafetyNet,
                 config: Configuration) -> None: # Assuming APIConfig is accessible via config or other components
        self.transactioncore = transactioncore
        self.marketmonitor = marketmonitor
        self.safetynet = safetynet
        # self.apiconfig = apiconfig # Removed, likely available via other components if needed
        self.config = config # Store main configuration
        self.strategy_types = ["eth_transaction", "front_run", "back_run", "sandwich_attack"]

        # Strategy registry maps type string to a list of async strategy functions
        self._strategy_registry: Dict[str, List[Callable[[Dict[str, Any]], Coroutine[Any, Any, Tuple[bool, Decimal]]]]] = {
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
        # A1 Update: Use the new dataclass for performance metrics
        self.strategy_performance: Dict[str, StrategyPerformanceMetrics] = {
            stype: StrategyPerformanceMetrics() for stype in self.strategy_types
        }

        # A4: Load weights on initialization
        self.weights_path = self._get_weights_path()
        self.reinforcement_weights = self._load_weights()

        # A1 Update: Use the new dataclass for configuration
        self.configuration: StrategyConfiguration = StrategyConfiguration(
            # Example of potentially loading from main config if needed
            # decay_factor=self.config.get_config_value("STRATEGY_DECAY_FACTOR", 0.95)
        )

        self._save_weights_task: Optional[asyncio.Task] = None
        self._is_shutting_down: bool = False

        # A2: Parameterized logging
        logger.debug("StrategyNet initialized.")

    def _get_weights_path(self) -> Path:
        """Determines the path for saving/loading RL weights."""
        runtime_dir = Path(self.config.get_config_value("RUNTIME_DIR", "./runtime"))
        runtime_dir.mkdir(parents=True, exist_ok=True)
        return runtime_dir / self.WEIGHTS_FILENAME

    def _load_weights(self) -> Dict[str, np.ndarray]:
        """Loads reinforcement weights from disk, initializing if not found."""
        weights: Dict[str, np.ndarray] = {}
        try:
            if self.weights_path.exists():
                loaded_data = np.load(str(self.weights_path), allow_pickle=True).item()
                # Validate loaded weights match current strategy structure
                for stype in self.strategy_types:
                    if stype in loaded_data and len(loaded_data[stype]) == len(self._strategy_registry.get(stype, [])):
                        weights[stype] = loaded_data[stype]
                    else:
                        logger.warning("Weight mismatch for strategy type '%s'. Reinitializing.", stype)
                        weights[stype] = np.ones(len(self._strategy_registry.get(stype, [])))
                logger.info("Reinforcement learning weights loaded from %s", self.weights_path)
            else:
                raise FileNotFoundError("Weights file not found.")
        except (FileNotFoundError, EOFError, ValueError) as e:
            logger.warning("Failed to load weights (%s). Initializing default weights.", e)
            weights = {
                stype: np.ones(len(self._strategy_registry.get(stype, [])))
                for stype in self.strategy_types if self._strategy_registry.get(stype)
            }
        return weights

    async def _save_weights_periodically(self) -> None:
        """Periodically saves the reinforcement weights to disk."""
        while not self._is_shutting_down:
            await asyncio.sleep(self.SAVE_INTERVAL_SECONDS)
            if not self._is_shutting_down: # Re-check after sleep
                await self._save_weights_atomic()

    async def _save_weights_atomic(self) -> None:
        """Saves weights atomically (temp file + rename)."""
        temp_path = self.weights_path.with_suffix(".tmp")
        try:
            # Use asyncio.to_thread for the blocking file I/O
            await asyncio.to_thread(np.save, str(temp_path), self.reinforcement_weights)
            await asyncio.to_thread(os.replace, str(temp_path), str(self.weights_path))
            logger.debug("Reinforcement weights saved to %s", self.weights_path)
        except Exception as e:
            logger.error("Failed to save reinforcement weights: %s", e, exc_info=True)
            # Attempt to clean up temp file if it exists
            if temp_path.exists():
                try:
                    await asyncio.to_thread(os.remove, str(temp_path))
                except OSError as rm_err:
                    logger.error("Failed to remove temporary weights file %s: %s", temp_path, rm_err)

    async def initialize(self) -> None:
        """Initializes StrategyNet and starts background tasks."""
        # A4: Start periodic saving task
        self._save_weights_task = asyncio.create_task(self._save_weights_periodically())
        logger.info("StrategyNet initialization complete. Weight persistence active.")

    async def stop(self) -> None:
        """Stops StrategyNet, cancels tasks, and saves weights."""
        logger.info("Stopping StrategyNet...")
        self._is_shutting_down = True
        if self._save_weights_task:
            self._save_weights_task.cancel()
            try:
                await self._save_weights_task
            except asyncio.CancelledError:
                logger.debug("Weight saving task cancelled.")
            except Exception as e:
                logger.error("Error during weight saving task shutdown: %s", e, exc_info=True)

        # A4: Final save on graceful shutdown
        logger.info("Performing final weight save...")
        await self._save_weights_atomic()
        logger.info("StrategyNet stopped.")


    def get_strategies(self, strategy_type: str) -> List[Callable[[Dict[str, Any]], Coroutine[Any, Any, Tuple[bool, Decimal]]]]:
        """Returns the list of strategy functions for a given type."""
        return self._strategy_registry.get(strategy_type, [])

    # A3: Replace blocking numpy.random.choice with non-blocking random.choices
    async def _select_best_strategy(self, strategies: List[Callable[[Dict[str, Any]], Coroutine[Any, Any, Tuple[bool, Decimal]]]], strategy_type: str) -> Callable[[Dict[str, Any]], Coroutine[Any, Any, Tuple[bool, Decimal]]]:
        """Selects the best strategy based on weights or explores randomly."""
        if not strategies:
            raise ValueError("No strategies provided for selection.")

        weights = self.reinforcement_weights.get(strategy_type)
        if weights is None or len(weights) != len(strategies):
             logger.error("Weight mismatch for strategy type '%s'. Using uniform distribution.", strategy_type)
             # Fallback to uniform random choice if weights are inconsistent
             return random.choice(strategies)

        if random.random() < self.configuration.exploration_rate:
            # A2: Parameterized logging
            logger.debug("Exploration: randomly selecting a strategy for type %s.", strategy_type)
            return random.choice(strategies) # random.choice is non-blocking

        # Use softmax for probability distribution (avoids issues with zero weights)
        # Subtract max for numerical stability
        max_weight = np.max(weights)
        exp_weights = np.exp(weights - max_weight)
        probabilities = exp_weights / np.sum(exp_weights)

        # Ensure probabilities sum to 1 (handle potential floating point inaccuracies)
        probabilities /= probabilities.sum()

        # Use random.choices (non-blocking) for weighted selection
        selected_strategy = random.choices(strategies, weights=probabilities, k=1)[0]
        selected_index = strategies.index(selected_strategy)

        # A2 + A16: Parameterized logging + structured logging potential (tx_hash context added later)
        logger.debug(
            "Selected strategy '%s' (type: %s, weight: %.4f, prob: %.4f).",
            selected_strategy.__name__,
            strategy_type,
            weights[selected_index],
            probabilities[selected_index],
            extra={"component": "StrategyNet"}
        )
        return selected_strategy

    def _calculate_reward(self, profit_decimal: Decimal, execution_time: float, success: bool) -> float:
        """Calculates the reward based on profit, time, and success."""
        # Use Decimal for profit calculation consistency
        base_reward = float(profit_decimal) if success else -0.1 # Penalize failures

        # Penalties (adjust factors as needed)
        time_penalty = 0.01 * execution_time # Penalize slow execution
        risk_penalty = 0.05 # Constant risk penalty per attempt (can be refined)

        # Optional: Penalty for very slow executions (worst-case scenarios)
        worst_case_threshold = 2.0 # seconds
        worst_case_penalty = 0.05 * max(0.0, execution_time - worst_case_threshold)

        total_reward = base_reward - time_penalty - risk_penalty - worst_case_penalty

        # A2: Parameterized logging
        logger.debug(
            "Reward computed: success=%s, profit=%.6f, time=%.4f -> base=%.4f, time_penalty=%.4f, risk_penalty=%.4f, wc_penalty=%.4f, total=%.4f",
            success,
            profit_decimal,
            execution_time,
            base_reward,
            time_penalty,
            risk_penalty,
            worst_case_penalty,
            total_reward,
            extra={"component": "StrategyNet"}
        )
        return total_reward

    async def _update_strategy_metrics(self,
                                       strategy_name: str,
                                       strategy_type: str,
                                       success: bool,
                                       profit_decimal: Decimal, # A11: Use Decimal profit
                                       execution_time: float) -> None:
        """Updates performance metrics and reinforcement weights for a strategy."""
        current_metrics = self.strategy_performance.get(strategy_type)
        if not current_metrics:
            logger.error("Metrics not found for strategy type %s", strategy_type)
            return

        # Update metrics using immutable dataclass pattern (create new instance)
        new_total_executions = current_metrics.total_executions + 1
        new_successes = current_metrics.successes + 1 if success else current_metrics.successes
        new_failures = current_metrics.failures + 1 if not success else current_metrics.failures
        new_total_profit = current_metrics.total_profit + profit_decimal if success else current_metrics.total_profit

        decay = self.configuration.decay_factor
        new_avg_execution_time = (current_metrics.avg_execution_time * decay +
                                  execution_time * (1 - decay))
        new_success_rate = new_successes / new_total_executions if new_total_executions > 0 else 0.0

        # A1 Update: Create a new metrics object
        updated_metrics = StrategyPerformanceMetrics(
            total_executions=new_total_executions,
            successes=new_successes,
            failures=new_failures,
            total_profit=new_total_profit,
            avg_execution_time=new_avg_execution_time,
            success_rate=new_success_rate
        )
        self.strategy_performance[strategy_type] = updated_metrics

        # Update reinforcement weights
        strategy_index = self.get_strategy_index(strategy_name, strategy_type)
        if strategy_index != -1:
            current_weight = self.reinforcement_weights[strategy_type][strategy_index]
            # A11: Use Decimal profit for reward calculation
            reward = self._calculate_reward(profit_decimal, execution_time, success)

            # Simple Q-learning like update (can be refined)
            # gamma = 0.9 # Discount factor (if considering future rewards)
            # For stateless reward, gamma can be 0 or next_max_q = 0
            next_max_q = 0 # Assuming immediate reward, no future state value used here
            learning_rate = self.configuration.learning_rate
            updated_weight = current_weight + learning_rate * (reward + next_max_q - current_weight)

            # Ensure weights don't become negative or too small
            self.reinforcement_weights[strategy_type][strategy_index] = max(0.01, updated_weight)

            # A2: Parameterized logging
            logger.debug(
                "Updated weight for %s (type: %s, index: %d) from %.4f to %.4f (reward: %.4f)",
                strategy_name, strategy_type, strategy_index, current_weight, updated_weight, reward,
                extra={"component": "StrategyNet"}
            )
        else:
            # A2: Parameterized logging
            logger.warning("Could not find index for strategy '%s' of type '%s' to update weights.", strategy_name, strategy_type)

        # A2: Parameterized logging
        logger.debug(
            "Strategy metrics for %s updated: executions=%d, success_rate=%.4f, avg_time=%.4f, total_profit=%.6f",
            strategy_type, updated_metrics.total_executions, updated_metrics.success_rate,
            updated_metrics.avg_execution_time, updated_metrics.total_profit,
            extra={"component": "StrategyNet"}
        )

    def get_strategy_index(self, strategy_name: str, strategy_type: str) -> int:
        """Finds the index of a strategy function within its type list."""
        strategies = self.get_strategies(strategy_type)
        for idx, strat_func in enumerate(strategies):
            if strat_func.__name__ == strategy_name:
                return idx
        # A2: Parameterized logging
        logger.warning("Strategy function '%s' not found in registry for type '%s'.", strategy_name, strategy_type)
        return -1

    # A11: Update to use profit returned by strategy
    async def execute_best_strategy(self, target_tx: Dict[str, Any], strategy_type: str) -> bool:
        """Selects, executes the best strategy, and updates metrics."""
        strategies = self.get_strategies(strategy_type)
        if not strategies:
            # A2: Parameterized logging
            logger.debug("No strategies available for type %s.", strategy_type)
            return False

        tx_hash = target_tx.get('tx_hash', 'N/A') # A16
        log_extra = {"component": "StrategyNet", "tx_hash": tx_hash} # A16

        try:
            selected_strategy = await self._select_best_strategy(strategies, strategy_type)
        except ValueError as e:
             logger.error("Failed to select strategy for type %s: %s", strategy_type, e, extra=log_extra)
             return False

        # A2: Parameterized logging
        logger.info(
            "Executing strategy '%s' for type '%s' on tx %s...",
            selected_strategy.__name__, strategy_type, tx_hash, extra=log_extra
        )
        start_time = time.monotonic()

        try:
            # A11: Strategy now returns (bool success, Decimal profit)
            success, profit_decimal = await selected_strategy(target_tx)
            execution_time = time.monotonic() - start_time

            # A16: Add tx_hash to log
            log_extra_result = {"component": "StrategyNet", "tx_hash": tx_hash, "strategy": selected_strategy.__name__}
            if success:
                # A2: Parameterized logging
                logger.info(
                    "Strategy '%s' succeeded in %.4f seconds. Profit: %s ETH.",
                    selected_strategy.__name__, execution_time, f"{profit_decimal:.8f}", extra=log_extra_result
                )
            else:
                # A2: Parameterized logging
                logger.warning(
                    "Strategy '%s' failed after %.4f seconds.",
                    selected_strategy.__name__, execution_time, extra=log_extra_result
                )

            # A11: Use returned profit for metrics update
            await self._update_strategy_metrics(
                selected_strategy.__name__,
                strategy_type,
                success,
                profit_decimal, # Pass the profit returned by the strategy
                execution_time
            )
            return success
        except Exception as e:
            execution_time = time.monotonic() - start_time
            # A16: Add tx_hash to log
            log_extra_error = {"component": "StrategyNet", "tx_hash": tx_hash, "strategy": selected_strategy.__name__}
            # A2: Parameterized logging
            logger.error(
                "Exception during execution of strategy '%s' (%.4f s): %s",
                selected_strategy.__name__, execution_time, e, exc_info=True, extra=log_extra_error
            )
            # Consider failed execution for metrics update
            await self._update_strategy_metrics(
                selected_strategy.__name__,
                strategy_type,
                False, # Mark as failure
                Decimal("0"), # Assume zero profit on exception
                execution_time
            )
            return False
