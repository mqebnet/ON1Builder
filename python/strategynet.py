# LICENSE: MIT // github.com/John0n1/ON1Builder

import asyncio
import time
import numpy as np
import random
from typing import Any, Dict, List, Optional, Callable
from decimal import Decimal

from apiconfig import APIConfig
from transactioncore import TransactionCore
from safetynet import SafetyNet
from marketmonitor import MarketMonitor

from loggingconfig import setup_logging
import logging

logger = setup_logging("StrategyNet", level=logging.INFO)


class StrategyNet:
    """
    Advanced strategy network for MEV operations. This class maintains a registry of strategies,
    selects the best one based on reinforcement learning weights, executes the chosen strategy, and
    updates performance metrics accordingly.
    """
    REWARD_BASE_MULTIPLIER: float = -0.1
    REWARD_TIME_PENALTY: float = -0.01

    def __init__(
        self,
        transactioncore: Optional[TransactionCore],
        marketmonitor: Optional[MarketMonitor],
        safetynet: Optional[SafetyNet],
        apiconfig: Optional[APIConfig],
    ) -> None:
        """
        Initialize the StrategyNet.
        """
        self.transactioncore = transactioncore
        self.marketmonitor = marketmonitor
        self.safetynet = safetynet
        self.apiconfig = apiconfig
        self.strategy_types = [
            "eth_transaction",
            "front_run",
            "back_run",
            "sandwich_attack"
        ]
        self._strategy_registry: Dict[str, List[Callable[[Dict[str, Any]], asyncio.Future]]] = {
            "eth_transaction": [
                self.high_value_eth_transfer
            ],
            "front_run": [
                self.aggressive_front_run,
                self.predictive_front_run,
                self.volatility_front_run,
                self.advanced_front_run
            ],
            "back_run": [
                self.price_dip_back_run,
                self.flashloan_back_run,
                self.high_volume_back_run,
                self.advanced_back_run
            ],
            "sandwich_attack": [
                self.flash_profit_sandwich,
                self.price_boost_sandwich,
                self.arbitrage_sandwich,
                self.advanced_sandwich_attack
            ]
        }
        self.strategy_performance: Dict[str, StrategyPerformanceMetrics] = {
            strategy_type: StrategyPerformanceMetrics()
            for strategy_type in self.strategy_types
        }
        self.reinforcement_weights: Dict[str, np.ndarray] = {
            strategy_type: np.ones(len(self.get_strategies(strategy_type)))
            for strategy_type in self.strategy_types
        }
        self.history_data: List[Dict[str, Any]] = []
        self.configuration: StrategyConfiguration = StrategyConfiguration()
        logger.debug("StrategyNet initialized with configuration")

    async def initialize(self) -> None:
        """
        Initialize performance metrics and reinforcement weights for all strategy types.
        """
        try:
            self.strategy_performance = {
                strategy_type: StrategyPerformanceMetrics()
                for strategy_type in self.strategy_types
            }
            self.reinforcement_weights = {
                strategy_type: np.ones(len(self.get_strategies(strategy_type)))
                for strategy_type in self.strategy_types
            }
            logger.debug("StrategyNet initialized successfully")
        except Exception as e:
            logger.critical(f"StrategyNet initialization failed: {e}")
            raise

    def register_strategy(
        self, strategy_type: str, strategy_func: Callable[[Dict[str, Any]], asyncio.Future]
    ) -> None:
        """
        Register a new strategy function under a specified strategy type.
        """
        if strategy_type not in self.strategy_types:
            logger.warning(f"Attempted to register unknown strategy type: {strategy_type}")
            return
        self._strategy_registry[strategy_type].append(strategy_func)
        self.reinforcement_weights[strategy_type] = np.ones(len(self._strategy_registry[strategy_type]))
        logger.debug(f"Registered new strategy '{strategy_func.__name__}' under '{strategy_type}'")

    def get_strategies(self, strategy_type: str) -> List[Callable[[Dict[str, Any]], asyncio.Future]]:
        """
        Retrieve the list of strategy functions for a given strategy type.
        """
        return self._strategy_registry.get(strategy_type, [])

    async def execute_best_strategy(
        self,
        target_tx: Dict[str, Any],
        strategy_type: str
    ) -> bool:
        """
        Select and execute the best strategy for the given transaction based on reinforcement weights.
        Updates performance metrics and reinforcement weights based on the outcome.
        """
        strategies = self.get_strategies(strategy_type)
        if not strategies:
            logger.debug(f"No strategies available for type: {strategy_type}")
            return False

        try:
            start_time = time.time()
            selected_strategy = await self._select_best_strategy(strategies, strategy_type)

            profit_before = await self.transactioncore.get_current_profit()
            success = await self.execute_strategy(selected_strategy, target_tx)
            profit_after = await self.transactioncore.get_current_profit()

            execution_time = time.time() - start_time
            profit_made = profit_after - profit_before

            await self._update_strategy_metrics(
                selected_strategy.__name__,
                strategy_type,
                success,
                profit_made,
                execution_time,
            )

            return success

        except StrategyExecutionError as e:
            logger.error(f"Strategy execution failed: {str(e)}", exc_info=True)
            return False
        except Exception as e:
            logger.exception(f"Unexpected error during strategy execution: {e}")
            return False

    async def _select_best_strategy(
        self, strategies: List[Callable[[Dict[str, Any]], asyncio.Future]], strategy_type: str
    ) -> Callable[[Dict[str, Any]], asyncio.Future]:
        """
        Select the best strategy for the given type using an exploration/exploitation approach.
        """
        weights = self.reinforcement_weights[strategy_type]
        if random.random() < self.configuration.exploration_rate:
            logger.debug("Using exploration for strategy selection")
            return random.choice(strategies)

        max_weight = np.max(weights)
        exp_weights = np.exp(weights - max_weight)
        probabilities = exp_weights / exp_weights.sum()
        selected_index = np.random.choice(len(strategies), p=probabilities)
        selected_strategy = strategies[selected_index]
        logger.debug(f"Selected strategy '{selected_strategy.__name__}' with weight {weights[selected_index]:.4f}")
        return selected_strategy

    async def _update_strategy_metrics(
        self,
        strategy_name: str,
        strategy_type: str,
        success: bool,
        profit: Decimal,
        execution_time: float,
    ) -> None:
        """
        Update performance metrics and reinforcement weights based on the strategy execution outcome.
        """
        metrics = self.strategy_performance[strategy_type]
        metrics.total_executions += 1

        if success:
            metrics.successes += 1
            metrics.profit += profit
        else:
            metrics.failures += 1

        metrics.avg_execution_time = (
            metrics.avg_execution_time * self.configuration.decay_factor
            + execution_time * (1 - self.configuration.decay_factor)
        )
        metrics.success_rate = metrics.successes / metrics.total_executions

        strategy_index = self.get_strategy_index(strategy_name, strategy_type)
        if strategy_index >= 0:
            reward = self._calculate_reward(success, profit, execution_time)
            self._update_reinforcement_weight(strategy_type, strategy_index, reward)

        self.history_data.append(
            {
                "timestamp": time.time(),
                "strategy_name": strategy_name,
                "success": success,
                "profit": float(profit),
                "execution_time": execution_time,
                "total_profit": float(metrics.profit),
            }
        )

    def get_strategy_index(self, strategy_name: str, strategy_type: str) -> int:
        """
        Get the index of a strategy function within its type.
        """
        strategies = self.get_strategies(strategy_type)
        for index, strategy in enumerate(strategies):
            if strategy.__name__ == strategy_name:
                return index
        logger.warning(f"Strategy '{strategy_name}' not found in type '{strategy_type}'")
        return -1

    def _calculate_reward(self, success: bool, profit: Decimal, execution_time: float) -> float:
        """
        Calculate a reward for a strategy based on execution outcome, profit, and time.
        """
        base_reward = float(profit) if success else self.REWARD_BASE_MULTIPLIER
        time_penalty = self.REWARD_TIME_PENALTY * execution_time
        total_reward = base_reward + time_penalty
        logger.debug(f"Calculated reward: {total_reward:.4f} (Base: {base_reward}, Time Penalty: {time_penalty})")
        return total_reward

    def _update_reinforcement_weight(self, strategy_type: str, index: int, reward: float) -> None:
        """
        Update the reinforcement weight of a given strategy.
        """
        lr = self.configuration.learning_rate
        current_weight = self.reinforcement_weights[strategy_type][index]
        new_weight = current_weight * (1 - lr) + reward * lr
        self.reinforcement_weights[strategy_type][index] = max(0.1, new_weight)
        logger.debug(f"Updated weight for strategy index {index} in '{strategy_type}': {new_weight:.4f}")

    async def high_value_eth_transfer(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute a high-value ETH transfer strategy if the transaction value exceeds a configurable threshold.
        """
        logger.info("Initiating High-Value ETH Transfer Strategy...")
        try:
            if not isinstance(target_tx, dict) or not target_tx:
                logger.debug("Invalid transaction format provided!")
                return False

            eth_value_in_wei = int(target_tx.get("value", 0))
            gas_price = int(target_tx.get("gasPrice", 0))
            to_address = target_tx.get("to", "")

            eth_value = self.transactioncore.web3.from_wei(eth_value_in_wei, "ether")
            gas_price_gwei = self.transactioncore.web3.from_wei(gas_price, "gwei")

            base_threshold = self.transactioncore.web3.to_wei(self.configuration.ETH_TRANSFER_BASE_THRESHOLD_ETH, "ether")
            if gas_price_gwei > self.configuration.ETH_TRANSFER_HIGH_GAS_PRICE_THRESHOLD_GWEI:
                threshold = base_threshold * 2
            elif gas_price_gwei > self.configuration.ETH_TRANSFER_MID_GAS_PRICE_THRESHOLD_GWEI:
                threshold = base_threshold * 1.5
            else:
                threshold = base_threshold

            threshold_eth = self.transactioncore.web3.from_wei(threshold, 'ether')
            logger.debug(
                f"Transaction Analysis:\n"
                f"Value: {eth_value:.4f} ETH\n"
                f"Gas Price: {gas_price_gwei:.2f} Gwei\n"
                f"To Address: {to_address[:10]}...\n"
                f"Current Threshold: {threshold_eth} ETH"
            )

            if eth_value <= 0:
                logger.debug("Transaction value is zero or negative. Skipping...")
                return False

            if not self.transactioncore.web3.is_address(to_address):
                logger.debug("Invalid recipient address. Skipping...")
                return False

            is_contract = await self.transactioncore._is_contract_address(to_address)
            if is_contract:
                logger.debug("Recipient is a contract. Additional validation required...")
                if not await self.transactioncore._validate_contract_interaction({"to": to_address}):
                    return False

            if eth_value_in_wei > threshold:
                logger.debug(
                    f"High-value ETH transfer detected:\n"
                    f"Value: {eth_value:.4f} ETH\n"
                    f"Threshold: {threshold_eth} ETH"
                )
                return await self.transactioncore.handle_eth_transaction(target_tx)

            logger.debug(
                f"ETH transaction value ({eth_value:.4f} ETH) below threshold "
                f"({threshold_eth} ETH). Skipping..."
            )
            return False

        except Exception as e:
            logger.error(f"Error in high-value ETH transfer strategy: {e}")
            return False

    async def aggressive_front_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute an aggressive front-run strategy based on risk assessment.
        """
        logger.debug("Initiating Aggressive Front-Run Strategy...")
        valid, decoded_tx, token_symbol = await self.transactioncore._validate_transaction(
            target_tx, "front_run", min_value=self.configuration.AGGRESSIVE_FRONT_RUN_MIN_VALUE_ETH 
        )
        if not valid:
            return False

        # Removed extra token_symbol parameter here. Now _calculate_risk_score expects only target_tx and price_change.
        risk_score, market_conditions = await self.transactioncore._calculate_risk_score(
            target_tx,
            price_change=await self.apiconfig.get_price_change_24h(token_symbol)
        )

        if risk_score >= self.configuration.AGGRESSIVE_FRONT_RUN_RISK_SCORE_THRESHOLD:
            logger.debug(f"Executing aggressive front-run (Risk: {risk_score:.2f})")
            return await self.transactioncore.front_run(target_tx)

        return False

    async def predictive_front_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute a predictive front-run strategy using price predictions and market data.
        """
        logger.debug("Initiating Predictive Front-Run Strategy...")
        valid, decoded_tx, token_symbol = await self.transactioncore._validate_transaction(target_tx, "front_run")
        if not valid:
            return False

        try:
            data = await asyncio.gather(
                self.marketmonitor.predict_price_movement(token_symbol),
                self.apiconfig.get_real_time_price(token_symbol),
                self.marketmonitor.check_market_conditions(target_tx["to"]),
                self.apiconfig.get_token_price_data(token_symbol, 'historical', timeframe=1),
                return_exceptions=True
            )
            predicted_price, current_price, market_conditions, historical_prices = data

            if any(isinstance(x, Exception) for x in data) or current_price is None or predicted_price is None:
                logger.warning("Failed to gather complete market data.")
                return False
        except Exception as e:
            logger.error(f"Error gathering market data: {e}")
            return False

        opportunity_score = await self.transactioncore._calculate_opportunity_score(
            price_change=(predicted_price / float(current_price) - 1) * 100,
            volatility=(np.std(historical_prices) / np.mean(historical_prices)) if historical_prices else 0,
            market_conditions=market_conditions,
            current_price=current_price,
            historical_prices=historical_prices
        )
        logger.debug(
            f"Predictive Analysis for {token_symbol}:\n"
            f"Current Price: {current_price:.6f}\n"
            f"Predicted Price: {predicted_price:.6f}\n"
            f"Expected Change: {(predicted_price / float(current_price) - 1) * 100:.2f}%\n"
            f"Opportunity Score: {opportunity_score}/100\n"
            f"Market Conditions: {market_conditions}"
        )
        if opportunity_score >= self.configuration.FRONT_RUN_OPPORTUNITY_SCORE_THRESHOLD:
            logger.debug(
                f"Executing predictive front-run for {token_symbol} "
                f"(Score: {opportunity_score}/100, Expected Change: {(predicted_price / float(current_price) - 1) * 100:.2f}%)"
            )
            return await self.transactioncore.front_run(target_tx)

        logger.debug(f"Opportunity score {opportunity_score}/100 below threshold. Skipping front-run.")
        return False

    async def volatility_front_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute a volatility-based front-run strategy using market volatility metrics.
        """
        logger.debug("Initiating Volatility Front-Run Strategy...")
        valid, decoded_tx, token_symbol = await self.transactioncore._validate_transaction(target_tx, "front_run")
        if not valid:
            return False

        try:
            results = await asyncio.gather(
                self.marketmonitor.check_market_conditions(target_tx["to"]),
                self.apiconfig.get_real_time_price(token_symbol),
                self.apiconfig.get_token_price_data(token_symbol, 'historical', timeframe=1),
                return_exceptions=True
            )
            market_conditions, current_price, historical_prices = results
            if any(isinstance(result, Exception) for result in results):
                logger.warning("Incomplete market data for volatility front-run")
                return False
        except Exception as e:
            logger.error(f"Error gathering market data: {e}")
            return False

        volatility_score = await self.transactioncore._calculate_volatility_score(
            historical_prices=historical_prices,
            current_price=current_price,
            market_conditions=market_conditions
        )

        logger.debug(
            f"Volatility Analysis for {token_symbol}:\n"
            f"Volatility Score: {volatility_score:.2f}/100\n"
            f"Current Price: {current_price}\n"
            f"24h Price Range: {min(historical_prices):.4f} - {max(historical_prices):.4f}\n"
            f"Market Conditions: {market_conditions}"
        )

        if volatility_score >= self.configuration.VOLATILITY_FRONT_RUN_SCORE_THRESHOLD:
            logger.debug(f"Executing volatility-based front-run for {token_symbol} (Volatility Score: {volatility_score:.2f}/100)")
            return await self.transactioncore.front_run(target_tx)

        logger.debug(f"Volatility score {volatility_score:.2f}/100 below threshold. Skipping front-run.")
        return False

    async def price_dip_back_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute a back-run strategy based on a predicted price dip.
        """
        logger.debug("Initiating Price Dip Back-Run Strategy...")
        valid, decoded_tx, token_symbol = await self.transactioncore._validate_transaction(target_tx, "back_run")
        if not valid:
            return False

        current_price = await self.apiconfig.get_real_time_price(token_symbol)
        if current_price is None:
            return False

        predicted_price = await self.marketmonitor.predict_price_movement(token_symbol)
        if predicted_price < float(current_price) * self.configuration.PRICE_DIP_BACK_RUN_THRESHOLD:
            logger.debug("Predicted price decrease meets threshold, proceeding with back-run.")
            return await self.transactioncore.back_run(target_tx)

        logger.debug("Predicted price decrease does not meet threshold. Skipping back-run.")
        return False

    async def flashloan_back_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute a flashloan-enabled back-run strategy.
        """
        logger.debug("Initiating Flashloan Back-Run Strategy...")
        estimated_amount = await self.transactioncore.calculate_flashloan_amount(target_tx)
        estimated_profit = estimated_amount * Decimal(str(self.configuration.FLASHLOAN_BACK_RUN_PROFIT_PERCENTAGE))
        if estimated_profit > self.configuration.MIN_PROFIT:
            logger.debug(f"Estimated profit: {estimated_profit} ETH meets threshold.")
            return await self.transactioncore.back_run(target_tx)
        logger.debug("Profit is insufficient for flashloan back-run. Skipping.")
        return False

    async def high_volume_back_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute a back-run strategy when high trading volume is detected.
        """
        logger.debug("Initiating High Volume Back-Run Strategy...")
        valid, decoded_tx, token_symbol = await self.transactioncore._validate_transaction(target_tx, "back_run")
        if not valid:
            return False

        volume_24h = await self.apiconfig.get_token_volume(token_symbol)
        volume_threshold = self._get_volume_threshold(token_symbol)
        if volume_24h > volume_threshold:
            logger.debug(f"High volume detected (${volume_24h:,.2f} USD), proceeding with back-run.")
            return await self.transactioncore.back_run(target_tx)

        logger.debug(f"Volume (${volume_24h:,.2f} USD) below threshold (${volume_threshold:,.2f} USD). Skipping.")
        return False

    async def advanced_back_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute an advanced back-run strategy based on favorable market conditions.
        """
        logger.debug("Initiating Advanced Back-Run Strategy...")
        valid, decoded_tx, token_symbol = await self.transactioncore._validate_transaction(target_tx, "back_run")
        if not valid:
            return False

        market_conditions = await self.marketmonitor.check_market_conditions(target_tx["to"])
        if market_conditions.get("high_volatility", False) and market_conditions.get("bullish_trend", False):
            logger.debug("Market conditions favorable for advanced back-run.")
            return await self.transactioncore.back_run(target_tx)
        logger.debug("Market conditions unfavorable for advanced back-run. Skipping.")
        return False

    async def flash_profit_sandwich(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute a sandwich attack strategy using flashloan profit estimation.
        """
        logger.debug("Initiating Flash Profit Sandwich Strategy...")
        estimated_amount = await self.transactioncore.calculate_flashloan_amount(target_tx)
        estimated_profit = estimated_amount * Decimal(str(self.configuration.FLASHLOAN_BACK_RUN_PROFIT_PERCENTAGE))
        gas_price = await self.safetynet.get_dynamic_gas_price()
        if gas_price > self.configuration.SANDWICH_ATTACK_GAS_PRICE_THRESHOLD_GWEI:
            logger.debug(f"Gas price too high for sandwich attack: {gas_price} Gwei")
            return False
        logger.debug(f"Executing sandwich with estimated profit: {estimated_profit:.4f} ETH")
        return await self.transactioncore.execute_sandwich_attack(target_tx)

    async def price_boost_sandwich(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute a sandwich attack strategy based on strong price momentum.
        """
        logger.debug("Initiating Price Boost Sandwich Strategy...")
        valid, decoded_tx, token_symbol = await self.transactioncore._validate_transaction(target_tx, "sandwich_attack")
        if not valid:
            return False

        historical_prices = await self.apiconfig.get_token_price_data(token_symbol, 'historical')
        if not historical_prices:
            logger.debug("No historical price data available, skipping price boost sandwich attack")
            return False

        momentum = await self.transactioncore._analyze_price_momentum(historical_prices)
        if momentum > self.configuration.PRICE_BOOST_SANDWICH_MOMENTUM_THRESHOLD:
            logger.debug(f"Strong price momentum detected: {momentum:.2%}")
            return await self.transactioncore.execute_sandwich_attack(target_tx)
        logger.debug(f"Insufficient price momentum: {momentum:.2%}. Skipping.")
        return False

    async def arbitrage_sandwich(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute a sandwich attack strategy based on arbitrage opportunities.
        """
        logger.debug("Initiating Arbitrage Sandwich Strategy...")
        valid, decoded_tx, token_symbol = await self.transactioncore._validate_transaction(target_tx, "sandwich_attack")
        if not valid:
            return False

        is_arbitrage = await self.marketmonitor._is_arbitrage_opportunity(target_tx)
        if is_arbitrage:
            logger.debug(f"Arbitrage opportunity detected for {token_symbol}")
            return await self.transactioncore.execute_sandwich_attack(target_tx)
        logger.debug("No profitable arbitrage opportunity found. Skipping.")
        return False

    async def advanced_sandwich_attack(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute an advanced sandwich attack strategy with integrated risk management.
        """
        logger.debug("Initiating Advanced Sandwich Attack...")
        valid, decoded_tx, token_symbol = await self.transactioncore._validate_transaction(target_tx, "sandwich_attack")
        if not valid:
            return False

        market_conditions = await self.marketmonitor.check_market_conditions(target_tx["to"])
        if market_conditions.get("high_volatility", False) and market_conditions.get("bullish_trend", False):
            logger.debug("Conditions favorable for sandwich attack.")
            return await self.transactioncore.execute_sandwich_attack(target_tx)
        logger.debug("Conditions unfavorable for sandwich attack. Skipping.")
        return False

    async def execute_strategy(
        self,
        strategy: Callable[[Dict[str, Any]], asyncio.Future],
        target_tx: Dict[str, Any]
    ) -> bool:
        """
        Execute a provided strategy function with the given transaction.
        """
        try:
            return await strategy(target_tx)
        except Exception as e:
            logger.error(f"Error executing strategy {strategy.__name__}: {e}")
            return False


class StrategyConfiguration:
    """
    Configuration settings for StrategyNet.
    """

    def __init__(self):
        self.decay_factor: float = 0.95
        self.min_profit_threshold: Decimal = Decimal("0.01")
        self.learning_rate: float = 0.01
        self.exploration_rate: float = 0.1
        self.RISK_ASSESSMENT_GAS_PRICE_THRESHOLD_GWEI: int = 300
        self.ETH_TRANSFER_HIGH_GAS_PRICE_THRESHOLD_GWEI: int = 200
        self.ETH_TRANSFER_MID_GAS_PRICE_THRESHOLD_GWEI: int = 100
        self.ETH_TRANSFER_BASE_THRESHOLD_ETH: int = 10
        self.FRONT_RUN_OPPORTUNITY_SCORE_THRESHOLD: int = 75
        self.VOLATILITY_FRONT_RUN_SCORE_THRESHOLD: int = 75
        self.ADVANCED_FRONT_RUN_RISK_SCORE_THRESHOLD: int = 75
        self.PRICE_DIP_BACK_RUN_THRESHOLD: float = 0.99
        self.FLASHLOAN_BACK_RUN_PROFIT_PERCENTAGE: float = 0.02
        self.HIGH_VOLUME_BACK_RUN_DEFAULT_THRESHOLD_USD: float = 100000
        self.SANDWICH_ATTACK_GAS_PRICE_THRESHOLD_GWEI: int = 200
        self.PRICE_BOOST_SANDWICH_MOMENTUM_THRESHOLD: float = 0.02
        self.AGGRESSIVE_FRONT_RUN_RISK_SCORE_THRESHOLD: float = 0.7
        self.AGGRESSIVE_FRONT_RUN_MIN_VALUE_ETH: float = 0.1


class StrategyPerformanceMetrics:
    """
    Tracks performance metrics for a strategy type.
    Now implemented with instance attributes.
    """
    def __init__(self):
        self.successes: int = 0
        self.failures: int = 0
        self.profit: Decimal = Decimal("0")
        self.avg_execution_time: float = 0.0
        self.success_rate: float = 0.0
        self.total_executions: int = 0


class StrategyExecutionError(Exception):
    """
    Exception raised when a strategy execution fails.
    """
    def __init__(self, message: str = "Strategy execution failed"):
        super().__init__(message)
