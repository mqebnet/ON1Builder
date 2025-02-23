#========================================================================================================================
# https://github.com/John0n1/0xBuilder

import asyncio
import time
import numpy as np
import random

from typing import Any, Dict, List, Optional, Callable, Tuple
from decimal import Decimal

from apiconfig import APIConfig
from transactioncore import TransactionCore 
from safetynet import SafetyNet  
from marketmonitor import MarketMonitor  

from loggingconfig import setup_logging
import logging
logger = setup_logging("StrategyNet", level=logging.INFO)


class StrategyNet:
    """Advanced strategy network for MEV operations."""

    REWARD_BASE_MULTIPLIER: float = -0.1
    REWARD_TIME_PENALTY: float = -0.01

    def __init__(
        self,
        transactioncore: Optional["TransactionCore"],
        marketmonitor: Optional["MarketMonitor"],
        safetynet: Optional["SafetyNet"],
        apiconfig: Optional["APIConfig"],
    ) -> None:
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

        self.strategy_performance: Dict[str, "StrategyPerformanceMetrics"] = {}
        self.reinforcement_weights: Dict[str, np.ndarray] = {}
        self.configuration: "StrategyConfiguration" = StrategyConfiguration()
        logger.debug("StrategyNet initialized with configuration")

    async def initialize(self) -> None:
        try:
            self.strategy_performance = {
                strategy_type: StrategyPerformanceMetrics()
                for strategy_type in self.strategy_types
            }
            self.reinforcement_weights = {
                strategy_type: np.ones(len(self.get_strategies(strategy_type)))
                for strategy_type in self.strategy_types
            }
            logger.debug("StrategyNet initialized ✅")
        except Exception as e:
            logger.critical(f"Strategy Net initialization failed: {e}")
            raise

    def register_strategy(self, strategy_type: str, strategy_func: Callable[[Dict[str, Any]], asyncio.Future]) -> None:
        if strategy_type not in self.strategy_types:
            logger.warning(f"Attempted to register unknown strategy type: {strategy_type}")
            return
        self._strategy_registry[strategy_type].append(strategy_func)
        self.reinforcement_weights[strategy_type] = np.ones(len(self._strategy_registry[strategy_type]))
        logger.debug(f"Registered new strategy '{strategy_func.__name__}' under '{strategy_type}'")

    def get_strategies(self, strategy_type: str) -> List[Callable[[Dict[str, Any]], asyncio.Future]]:
        return self._strategy_registry.get(strategy_type, [])

    async def execute_best_strategy(
        self,
        target_tx: Dict[str, Any],
        strategy_type: str
    ) -> bool:
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
        strategies = self.get_strategies(strategy_type)
        for index, strategy in enumerate(strategies):
            if strategy.__name__ == strategy_name:
                return index
        logger.warning(f"Strategy '{strategy_name}' not found in type '{strategy_type}'")
        return -1

    def _calculate_reward(
        self, success: bool, profit: Decimal, execution_time: float
    ) -> float:
        base_reward = float(profit) if success else self.REWARD_BASE_MULTIPLIER
        time_penalty = self.REWARD_TIME_PENALTY * execution_time
        total_reward = base_reward + time_penalty
        logger.debug(f"Calculated reward: {total_reward:.4f} (Base: {base_reward}, Time Penalty: {time_penalty})")
        return total_reward

    def _update_reinforcement_weight(
        self, strategy_type: str, index: int, reward: float
    ) -> None:
        lr = self.configuration.learning_rate
        current_weight = self.reinforcement_weights[strategy_type][index]
        new_weight = current_weight * (1 - lr) + reward * lr
        self.redebugrcement_weights[strategy_type][index] = max(0.1, new_weight)
        logger.debug(f"Updated weight for strategy index {index} in '{strategy_type}': {new_weight:.4f}")

    async def _decode_transaction(self, target_tx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            decoded = await self.transactioncore.decode_transaction_input(
                target_tx["input"], target_tx["to"]
            )
            logger.debug(f"Decoded transaction: {decoded}")
            return decoded
        except Exception as e:
            logger.error(f"Error decoding transaction: {e}")
            return None

    async def _get_token_symbol(self, token_address: str) -> Optional[str]:
        try:
            symbol = await self.apiconfig.get_token_symbol(
                self.transactioncore.web3, token_address
            )
            logger.debug(f"Retrieved token symbol '{symbol}' for address '{token_address}'")
            return symbol
        except Exception as e:
            logger.error(f"Error fetching token symbol: {e}")
            return None

    async def _assess_risk(
        self,
        tx: Dict[str, Any],
        token_symbol: str,
        price_change: float = 0,
        volume: float = 0
    ) -> Tuple[float, Dict[str, Any]]:
        try:
            risk_score = 1.0
            market_conditions = await self.marketmonitor.check_market_conditions(tx.get("to", ""))

            gas_price = int(tx.get("gasPrice", 0))
            gas_price_gwei = float(self.transactioncore.web3.from_wei(gas_price, "gwei"))
            if gas_price_gwei > self.configuration.MAX_GAS_PRICE_GWEI:
                risk_score *= 0.7

            if market_conditions.get("high_volatility", False):
                risk_score *= 0.7
            if market_conditions.get("low_liquidity", False):
                risk_score *= 0.6
            if market_conditions.get("bullish_trend", False):
                risk_score *= 1.2

            if price_change > 0:
                risk_score *= min(1.3, 1 + (price_change / 100))

            if volume >= 1_000_000:
                risk_score *= 1.2
            elif volume <= 100_000:
                risk_score *= 0.8

            risk_score = max(0.0, min(1.0, risk_score))
            logger.debug(f"Risk assessment for {token_symbol}: {risk_score:.2f}")

            return risk_score, market_conditions

        except Exception as e:
            logger.error(f"Error in risk assessment: {e}")
            return 0.0, {}

    async def high_value_eth_transfer(self, target_tx: Dict[str, Any]) -> bool:
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
                if not await self.transactioncore._validate_contract_interaction(to_address):
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
        logger.debug("Initiating Aggressive Front-Run Strategy...")

        valid, decoded_tx, token_symbol = await self.transactioncore._validate_transaction(
            target_tx, "front_run", min_value=self.configuration.AGGRESSIVE_FRONT_RUN_MIN_VALUE_ETH
        )
        if not valid:
            return False

        risk_score, market_conditions = await self.transactioncore._calculate_risk_score(
            target_tx,
            token_symbol,
            price_change=await self.apiconfig.get_price_change_24h(token_symbol)
        )

        if risk_score >= self.configuration.AGGRESSIVE_FRONT_RUN_RISK_SCORE_THRESHOLD:
            logger.debug(f"Executing aggressive front-run (Risk: {risk_score:.2f})")
            return await self.transactioncore.front_run(target_tx)

        return False

    async def predictive_front_run(self, target_tx: Dict[str, Any]) -> bool:
        logger.debug("Initiating  Predictive Front-Run Strategy...")

        valid, decoded_tx, token_symbol = await self.transactioncore._validate_transaction(
            target_tx, "front_run"
        )
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

            if any(isinstance(x, Exception) for x in data):
                logger.warning("Failed to gather complete market data.")
                return False

            if current_price is None or predicted_price is None:
                logger.debug("Missing price data for analysis.")
                return False

        except Exception as e:
            logger.error(f"Error gathering market data: {e}")
            return False

        opportunity_score = await self.transactioncore._calculate_opportunity_score(
            price_change= (predicted_price / float(current_price) - 1) * 100,
            volatility=np.std(historical_prices) / np.mean(historical_prices) if historical_prices else 0,
            market_conditions=market_conditions,
            current_price=current_price,
            historical_prices=historical_prices
        )

        logger.debug(
            f"Predictive Analysis for {token_symbol}:\n"
            f"Current Price: {current_price:.6f}\n"
            f"Predicted Price: {predicted_price:.6f}\n"
            f"Expected Change: {(predicted_price / float(current_price) - 1) * 100:.2f}%\n"
            f"Volatility: {np.std(historical_prices) / np.mean(historical_prices) if historical_prices else 0:.2f}\n"
            f"Opportunity Score: {opportunity_score}/100\n"
            f"Market Conditions: {market_conditions}"
        )

        if opportunity_score >= self.configuration.FRONT_RUN_OPPORTUNITY_SCORE_THRESHOLD:
            logger.debug(
                f"Executing predictive front-run for {token_symbol} "
                f"(Score: {opportunity_score}/100, Expected Change: {(predicted_price / float(current_price) - 1) * 100:.2f}%)"
            )
            return await self.transactioncore.front_run(target_tx)

        logger.debug(
            f"Opportunity score {opportunity_score}/100 below threshold. Skipping front-run."
        )
        return False

    async def volatility_front_run(self, target_tx: Dict[str, Any]) -> bool:
        logger.debug("Initiating  Volatility Front-Run Strategy...")

        valid, decoded_tx, token_symbol = await self.transactioncore._validate_transaction(
            target_tx, "front_run"
        )
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
                logger.warning("Failed to gather complete market data")
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
            logger.debug(
                f"Executing volatility-based front-run for {token_symbol} "
                f"(Volatility Score: {volatility_score:.2f}/100)"
            )
            return await self.transactioncore.front_run(target_tx)

        logger.debug(
            f"Volatility score {volatility_score:.2f}/100 below threshold. Skipping front-run."
        )
        return False

    async def advanced_front_run(self, target_tx: Dict[str, Any]) -> bool:
        logger.debug("Initiating Advanced Front-Run Strategy...")

        valid, decoded_tx, token_symbol = await self.transactioncore._validate_transaction(
            target_tx, "front_run"
        )
        if not valid:
            return False

        try:
             analysis_results = await asyncio.gather(
                 self.marketmonitor.predict_price_movement(token_symbol),
                 self.marketmonitor.check_market_conditions(target_tx["to"]),
                 self.apiconfig.get_real_time_price(token_symbol),
                self.apiconfig.get_token_volume(token_symbol),
                return_exceptions=True
             )

             predicted_price, market_conditions, current_price, volume = analysis_results

             if any(isinstance(result, Exception) for result in analysis_results):
                 logger.warning("Failed to gather complete market data.")
                 return False

             if current_price is None or predicted_price is None:
                 logger.debug("Missing price data for analysis. Skipping...")
                 return False

        except Exception as e:
            logger.error(f"Error gathering market data: {e}")
            return False

        price_increase = (predicted_price / float(current_price) - 1) * 100
        is_bullish = market_conditions.get("bullish_trend", False)
        is_volatile = market_conditions.get("high_volatility", False)
        has_liquidity = not market_conditions.get("low_liquidity", True)

        risk_score, market_conditions = await self.transactioncore._calculate_risk_score(
            target_tx,
            token_symbol,
            price_change=price_increase
        )

        logger.debug(
            f"Analysis for {token_symbol}:\n"
            f"Price Increase: {price_increase:.2f}%\n"
            f"Market Trend: {'Bullish' if is_bullish else 'Bearish'}\n"
            f"Volatility: {'High' if is_volatile else 'Low'}\n"
            f"Liquidity: {'Adequate' if has_liquidity else 'Low'}\n"
            f"24h Volume: ${volume:,.2f}\n"
            f"Risk Score: {risk_score}/100"
        )

        if risk_score >= self.configuration.ADVANCED_FRONT_RUN_RISK_SCORE_THRESHOLD:
            logger.debug(
                f"Executing advanced front-run for {token_symbol} "
                f"(Risk Score: {risk_score}/100)"
            )
            return await self.transactioncore.front_run(target_tx)

        logger.debug(
            f"Risk score {risk_score}/100 below threshold. Skipping front-run."
        )
        return False

    async def price_dip_back_run(self, target_tx: Dict[str, Any]) -> bool:
        logger.debug("Initiating Price Dip Back-Run Strategy...")

        valid, decoded_tx, token_symbol = await self.transactioncore._validate_transaction(
            target_tx, "back_run"
        )
        if not valid:
            return False

        current_price = await self.apiconfig.get_real_time_price(token_symbol)
        if current_price is None:
            return False

        predicted_price = await self.marketmonitor.predict_price_movement(token_symbol)
        if predicted_price < float(current_price) * self.configuration.PRICE_DIP_BACK_RUN_THRESHOLD:
            logger.debug("Predicted price decrease exceeds threshold, proceeding with back-run.")
            return await self.transactioncore.back_run(target_tx)

        logger.debug("Predicted price decrease does not meet threshold. Skipping back-run.")
        return False

    async def flashloan_back_run(self, target_tx: Dict[str, Any]) -> bool:
        logger.debug("Initiating Flashloan Back-Run Strategy...")
        estimated_amount = await self.transactioncore.calculate_flashloan_amount(target_tx)
        estimated_profit = estimated_amount * Decimal(str(self.configuration.FLASHLOAN_BACK_RUN_PROFIT_PERCENTAGE))
        if estimated_profit > self.configuration.min_profit_threshold:
            logger.debug(f"Estimated profit: {estimated_profit} ETH meets threshold.")
            return await self.transactioncore.back_run(target_tx)
        logger.debug("Profit is insufficient for flashloan back-run. Skipping.")
        return False

    async def high_volume_back_run(self, target_tx: Dict[str, Any]) -> bool:
        logger.debug("Initiating High Volume Back-Run Strategy...")

        valid, decoded_tx, token_symbol = await self.transactioncore._validate_transaction(
            target_tx, "back_run"
        )
        if not valid:
            return False

        volume_24h = await self.apiconfig.get_token_volume(token_symbol)
        volume_threshold = await self.transactioncore._get_volume_threshold(token_symbol)
        if volume_24h > volume_threshold:
            logger.debug(f"High volume detected (${volume_24h:,.2f} USD), proceeding with back-run.")
            return await self.transactioncore.back_run(target_tx)

        logger.debug(f"Volume (${volume_24h:,.2f} USD) below threshold (${volume_threshold:,.2f} USD). Skipping.")
        return False

    async def advanced_back_run(self, target_tx: Dict[str, Any]) -> bool:
        logger.debug("Initiating Advanced Back-Run Strategy...")

        valid, decoded_tx, token_symbol = await self.transactioncore._validate_transaction(
            target_tx, "back_run"
        )
        if not valid:
            return False

        market_conditions = await self.marketmonitor.check_market_conditions(
            target_tx["to"]
        )
        if market_conditions.get("high_volatility", False) and market_conditions.get(
            "bullish_trend", False
        ):
            logger.debug("Market conditions favorable for advanced back-run.")
            return await self.transactioncore.back_run(target_tx)

        logger.debug("Market conditions unfavorable for advanced back-run. Skipping.")
        return False

    async def flash_profit_sandwich(self, target_tx: Dict[str, Any]) -> bool:
        logger.debug("Initiating Flash Profit Sandwich Strategy...")
        estimated_amount = await self.transactioncore.calculate_flashloan_amount(target_tx)
        estimated_profit = estimated_amount * Decimal(str(self.configuration.FLASHLOAN_BACK_RUN_PROFIT_PERCENTAGE)) 
        gas_price = await self.transactioncore.get_dynamic_gas_price()
        if (gas_price > self.configuration.SANDWICH_ATTACK_GAS_PRICE_THRESHOLD_GWEI): 
            logger.debug(f"Gas price too high for sandwich attack: {gas_price} Gwei")
            return False
        logger.debug(f"Executing sandwich with estimated profit: {estimated_profit:.4f} ETH")
        return await self.transactioncore.execute_sandwich_attack(target_tx)

    async def price_boost_sandwich(self, target_tx: Dict[str, Any]) -> bool:
        logger.debug("Initiating Price Boost Sandwich Strategy...")

        valid, decoded_tx, token_symbol = await self.transactioncore._validate_transaction(
            target_tx, "sandwich_attack"
        )
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
        logger.debug("Initiating Arbitrage Sandwich Strategy...")

        valid, decoded_tx, token_symbol = await self.transactioncore._validate_transaction(
            target_tx, "sandwich_attack"
        )
        if not valid:
            return False

        is_arbitrage = await self.marketmonitor.is_arbitrage_opportunity(target_tx)
        if is_arbitrage:
            logger.debug(f"Arbitrage opportunity detected for {token_symbol}")
            return await self.transactioncore.execute_sandwich_attack(target_tx)

        logger.debug("No profitable arbitrage opportunity found. Skipping.")
        return False

    async def advanced_sandwich_attack(self, target_tx: Dict[str, Any]) -> bool:
        logger.debug("Initiating Advanced Sandwich Attack...")

        valid, decoded_tx, token_symbol = await self.transactioncore._validate_transaction(
            target_tx, "sandwich_attack"
        )
        if not valid:
            return False

        market_conditions = await self.marketmonitor.check_market_conditions(
            target_tx["to"]
        )
        if market_conditions.get("high_volatility", False) and market_conditions.get(
            "bullish_trend", False
        ):
            logger.debug("Conditions favorable for sandwich attack.")
            return await self.transactioncore.execute_sandwich_attack(target_tx)

        logger.debug("Conditions unfavorable for sandwich attack. Skipping.")
        return False

    async def stop(self) -> None:
        try:
            self.strategy_performance.clear()
            self.reinforcement_weights.clear()
            self.history_data.clear()
            logger.info("Strategy Net stopped successfully.")
        except Exception as e:
            logger.error(f"Error stopping Strategy Net: {e}")

    async def _estimate_profit(self, tx: Any, decoded_params: Dict[str, Any]) -> Decimal:
        try:
            path = decoded_params.get('path', [])
            value = getattr(tx, 'value', 0)
            gas_price = getattr(tx, 'gasPrice', 0)

            estimated_profit = await self.transactioncore.estimate_transaction_profit(
                tx, path, value, gas_price
            )
            logger.debug(f"Estimated profit: {estimated_profit:.4f} ETH")
            return estimated_profit
        except Exception as e:
            logger.error(f"Error estimating profit: {e}")
            return Decimal("0")

    async def execute_strategy(
        self,
        strategy: Callable[[Dict[str, Any]], asyncio.Future],
        target_tx: Dict[str, Any]
    ) -> bool:
        try:
            return await strategy(target_tx)
        except Exception as e:
            logger.error(f"Error executing strategy {strategy.__name__}: {e}")
            return False

class StrategyConfiguration:
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
    successes: int = 0
    failures: int = 0
    profit: Decimal = Decimal("0")
    avg_execution_time: float = 0.0
    success_rate: float = 0.0
    total_executions: int = 0

class StrategyExecutionError(Exception):
    def __init__(self, message: str = "Strategy execution failed"):
        super().__init__(message)
