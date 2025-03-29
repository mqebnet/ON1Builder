#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MempoolMonitor Module

This module monitors pending transactions from the mempool, queues them for processing based on priority,
and analyzes transactions for profitability. It supports both filter-based and polling-based monitoring
mechanisms and integrates with the SafetyNet, NonceCore, APIConfig, and MarketMonitor components.
"""

import asyncio
import time
import async_timeout
import numpy as np
from decimal import Decimal
from typing import Any, Dict, List, Optional

from web3 import AsyncWeb3
from web3.exceptions import TransactionNotFound, Web3ValueError

from abiregistry import ABIRegistry
from apiconfig import APIConfig
from configuration import Configuration
from marketmonitor import MarketMonitor
from noncecore import NonceCore
from safetynet import SafetyNet

from loggingconfig import setup_logging
import logging

logger = setup_logging("MempoolMonitor", level=logging.INFO)


class MempoolMonitor:
    """
    Monitors the mempool for pending transactions, prioritizes them,
    and processes each transaction to identify profitable opportunities.
    """

    def __init__(
        self,
        web3: AsyncWeb3,
        safetynet: SafetyNet,
        noncecore: NonceCore,
        apiconfig: APIConfig,
        monitored_tokens: Optional[List[str]] = None,
        configuration: Optional[Configuration] = None,
        erc20_abi: Optional[List[Dict[str, Any]]] = None,
        marketmonitor: Optional[MarketMonitor] = None,
    ):
        """
        Initialize the MempoolMonitor.
        """
        self.web3 = web3
        self.configuration = configuration
        self.safetynet = safetynet
        self.noncecore = noncecore
        self.apiconfig = apiconfig
        self.marketmonitor = marketmonitor

        self.running = False
        self.pending_transactions = asyncio.Queue()
        self.monitored_tokens = set(monitored_tokens or [])
        self.processed_transactions_lock = asyncio.Lock()
        self.profitable_transactions = asyncio.Queue()
        self.processed_transactions = set()

        if not erc20_abi or not isinstance(erc20_abi, list):
            logger.error("Invalid or missing ERC20 ABI")
            self.erc20_abi = []
        else:
            self.erc20_abi = erc20_abi
            logger.debug(f"Loaded ERC20 ABI with {len(self.erc20_abi)} entries")

        self.minimum_profit_threshold = Decimal("0.001")
        self.backoff_factor = 1.5
        # Replace regular Queue with a PriorityQueue for task ordering.
        self.task_queue = asyncio.PriorityQueue()
        # Predefined function selectors (can be updated from config)
        self.function_signatures = {
            '0xa9059cbb': 'transfer',
            '0x095ea7b3': 'approve',
            '0x23b872dd': 'transferFrom'
        }

        logger.debug("Starting main engine...")
        time.sleep(1)

        self.abiregistry = ABIRegistry()
        self.cache = {}

    async def initialize(self) -> None:
        """
        Initialize the MempoolMonitor by updating function signatures,
        validating the ERC20 ABI, and resetting queues and caches.
        
        Raises:
            ValueError: If the ERC20 ABI is missing or invalid.
        """
        try:
            logger.debug("Initializing MempoolMonitor...")
            if self.configuration and hasattr(self.configuration, 'get_erc20_signatures'):
                self.function_signatures.update(await self.configuration.get_erc20_signatures())
            if not self.erc20_abi:
                raise ValueError("Failed to load ERC20 ABI")
            if not self.abiregistry._validate_abi(self.erc20_abi, 'erc20'):
                raise ValueError("Invalid ERC20 ABI")
            self.running = False
            self.pending_transactions = asyncio.Queue()
            self.profitable_transactions = asyncio.Queue()
            self.processed_transactions = set()
            self.task_queue = asyncio.PriorityQueue()
            logger.debug("MempoolMonitor initialized")
        except Exception as e:
            logger.critical(f"MempoolMonitor initialization failed: {e}")
            raise

    async def start_monitoring(self) -> None:
        """
        Start monitoring the mempool for pending transactions.
        Creates background tasks for monitoring and processing the task queue.
        """
        if self.running:
            logger.debug("Monitoring already active.")
            return
        try:
            self.running = True
            monitoring_task = asyncio.create_task(self._run_monitoring())
            processor_task = asyncio.create_task(self._process_task_queue())
            logger.info("Mempool monitoring started")
            await asyncio.gather(monitoring_task, processor_task)
        except Exception as e:
            self.running = False
            logger.error(f"Error starting monitoring: {e}")

    async def _run_monitoring(self) -> None:
        """
        Run the monitoring loop using either filter-based or polling-based mechanisms.
        Implements a circuit breaker for repeated errors.
        """
        retry_count = 0
        error_count = 0
        while self.running:
            try:
                try:
                    pending_filter = await self._setup_pending_filter()
                    if pending_filter:
                        while self.running:
                            tx_hashes = await pending_filter.get_new_entries()
                            await self._handle_new_transactions(tx_hashes)
                            await asyncio.sleep(1)
                    else:
                        await self._poll_pending_transactions()
                except Exception as filter_error:
                    logger.warning(f"Filter-based monitoring failed: {filter_error}")
                    logger.debug("Switching to polling-based monitoring...")
                    await self._poll_pending_transactions()
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                retry_count += 1
                error_count += 1
                if error_count >= self.configuration.CIRCUIT_BREAKER_THRESHOLD:
                    logger.error("Circuit breaker activated. Halting monitoring temporarily.")
                    await asyncio.sleep(self.configuration.CIRCUIT_BREAKER_DELAY)
                    error_count = 0
                await asyncio.sleep(self.configuration.MEMPOOL_RETRY_DELAY * retry_count)

    async def _poll_pending_transactions(self) -> None:
        """
        Poll new blocks for transactions when filter-based monitoring is unavailable.
        """
        last_block = await self.web3.eth.block_number
        while self.running:
            try:
                current_block = await self.web3.eth.block_number
                if current_block <= last_block:
                    await asyncio.sleep(1)
                    continue
                for block_num in range(last_block + 1, current_block + 1):
                    try:
                        block = await self.web3.eth.get_block(block_num, full_transactions=True)
                        if block and block.transactions:
                            tx_hashes = [
                                tx.hash.hex() if hasattr(tx, 'hash') else tx['hash'].hex()
                                for tx in block.transactions
                            ]
                            await self._handle_new_transactions(tx_hashes)
                    except Exception as e:
                        logger.error(f"Error processing block {block_num}: {e}")
                        continue
                last_block = current_block
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error in polling loop: {e}")
                await asyncio.sleep(2)

    async def _setup_pending_filter(self) -> Optional[Any]:
        """
        Attempt to set up a pending transaction filter.
        """
        try:
            pending_filter = await self.web3.eth.filter("pending")
            try:
                async with async_timeout.timeout(5):
                    await pending_filter.get_new_entries()
                    logger.debug("Successfully set up pending transaction filter")
                    return pending_filter
            except asyncio.TimeoutError:
                logger.warning("Filter setup timed out, falling back to polling")
                return None
            except Exception as e:
                logger.warning(f"Filter validation failed: {e}, retrying after delay")
                await asyncio.sleep(self.configuration.MEMPOOL_RETRY_DELAY)
                return None
        except Exception as e:
            logger.warning(f"Failed to setup pending filter: {e}, falling back to polling")
            return None

    async def _handle_new_transactions(self, tx_hashes: List[str]) -> None:
        """
        Process a list of new transaction hashes in batches.
        """
        async def process_batch(batch: List[str]) -> None:
            await asyncio.gather(*(self._queue_transaction(tx_hash) for tx_hash in batch))
        try:
            for i in range(0, len(tx_hashes), self.configuration.MEMPOOL_BATCH_SIZE):
                batch = tx_hashes[i : i + self.configuration.MEMPOOL_BATCH_SIZE]
                await process_batch(batch)
        except Exception as e:
            logger.error(f"Error handling new transactions: {e}")

    async def _queue_transaction(self, tx_hash: str) -> None:
        """
        Queue a transaction for processing if it has not been processed already.
        Priority is determined by the transaction's gas price.
        """
        if not tx_hash:
            logger.warning("Invalid transaction hash received")
            return
        tx_hash_hex = tx_hash.hex() if isinstance(tx_hash, bytes) else tx_hash
        async with self.processed_transactions_lock:
            if tx_hash_hex not in self.processed_transactions:
                self.processed_transactions.add(tx_hash_hex)
                priority = await self._calculate_priority(tx_hash_hex)
                # Use the PriorityQueue's put method to store tuple (priority, tx_hash)
                await self.task_queue.put((priority, tx_hash_hex))

    async def _calculate_priority(self, tx_hash: str) -> int:
        """
        Calculate the priority for a transaction based on its gas price.
        Higher gas price yields higher priority (lower numerical value).
        """
        try:
            tx = await self._get_transaction_with_retry(tx_hash)
            if not tx:
                return float('inf')
            gas_price = tx.get('gasPrice', 0)
            return -gas_price
        except Exception as e:
            logger.error(f"Error calculating priority for transaction {tx_hash}: {e}")
            return float('inf')

    async def _process_task_queue(self) -> None:
        """
        Continuously process transactions from the task queue using a semaphore to limit parallelism.
        """
        while self.running:
            try:
                # Retrieve the next item from the PriorityQueue
                priority, tx_hash = await self.task_queue.get()
                async with asyncio.Semaphore(self.configuration.MEMPOOL_MAX_PARALLEL_TASKS):
                    await self.process_transaction(tx_hash)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing task queue: {e}")

    async def process_transaction(self, tx_hash: str) -> None:
        """
        Process an individual transaction: fetch its details, analyze its profitability,
        and if profitable, handle it accordingly.
        """
        try:
            tx = await self._get_transaction_with_retry(tx_hash)
            if not tx:
                return
            analysis = await self.analyze_transaction(tx)
            if analysis.get("is_profitable"):
                await self._handle_profitable_transaction(analysis)
        except Exception as e:
            logger.debug(f"Error processing transaction {tx_hash}: {e}")

    async def _get_transaction_with_retry(self, tx_hash: str) -> Optional[Any]:
        """
        Retrieve a transaction with retries and exponential backoff.
        """
        backoff = self.configuration.MEMPOOL_RETRY_DELAY
        for attempt in range(self.configuration.MEMPOOL_MAX_RETRIES):
            try:
                if tx_hash in self.cache:
                    return self.cache[tx_hash]
                tx = await self.web3.eth.get_transaction(tx_hash)
                self.cache[tx_hash] = tx
                return tx
            except TransactionNotFound:
                if attempt == self.configuration.MEMPOOL_MAX_RETRIES - 1:
                    return None
                await asyncio.sleep(backoff)
                backoff *= self.backoff_factor
            except Exception as e:
                error_str = str(e)
                if "indexing is in progress" in error_str:
                    if attempt < self.configuration.MEMPOOL_MAX_RETRIES - 1:
                        await asyncio.sleep(backoff)
                        backoff *= self.backoff_factor
                        continue
                logger.error(f"Error fetching transaction {tx_hash}: {e}")
                return None

    async def _handle_profitable_transaction(self, analysis: Dict[str, Any]) -> None:
        """
        Handle a profitable transaction by logging details, tagging it with a strategy type,
        and queuing it for further processing.
        """
        try:
            profit = analysis.get('profit', Decimal(0))
            if isinstance(profit, (int, float)):
                profit = Decimal(str(profit))
            elif not isinstance(profit, Decimal):
                logger.warning(f"Invalid profit type: {type(profit)}")
                profit = Decimal(0)
            profit_str = f"{float(profit):.6f}" if profit > 0 else 'Unknown'
            analysis['profit'] = profit
            analysis['timestamp'] = time.time()
            analysis['gas_price'] = self.web3.from_wei(analysis.get('gasPrice', 0), 'gwei')
            analysis['strategy_type'] = self._determine_strategy_type(analysis)
            await self.profitable_transactions.put(analysis)
            logger.debug(
                f"Profitable transaction identified: {analysis['tx_hash']} "
                f"(Estimated profit: {profit_str} ETH, Strategy Type: {analysis['strategy_type']})"
            )
        except Exception as e:
            logger.error(f"Error handling profitable transaction: {e}")

    def _determine_strategy_type(self, analysis: Dict[str, Any]) -> str:
        """
        Determine the strategy type for a transaction based on its analysis.
        """
        if analysis.get('value', 0) > 0 and 'input' not in analysis:
            return "eth_transaction"
        if analysis.get('function_name') in ("swap", "swapExactTokensForTokens", "swapTokensForExactTokens"):
            if "amountOutMin" in analysis.get('params', {}):
                if 'path' in analysis.get('params', {}):
                    return "sandwich_attack"
                return "back_run"
            return "front_run"
        return "unknown"

    async def analyze_transaction(self, tx) -> Dict[str, Any]:
        """
        Analyze a transaction to determine its profitability.
        """
        if not tx.hash or not tx.input:
            logger.debug(f"Transaction {tx.hash.hex()} is missing essential fields. Skipping.")
            return {"is_profitable": False}
        try:
            if tx.value > 0:
                return await self._analyze_eth_transaction(tx)
            return await self._analyze_token_transaction(tx)
        except Exception as e:
            logger.error(f"Error analyzing transaction {tx.hash.hex()}: {e}")
            return {"is_profitable": False}

    async def _analyze_eth_transaction(self, tx) -> Dict[str, Any]:
        """
        Analyze an ETH transaction for profitability.
        """
        try:
            if await self._is_profitable_eth_transaction(tx):
                await self._log_transaction_details(tx, is_eth=True)
                return {
                    "is_profitable": True,
                    "tx_hash": tx.hash.hex(),
                    "value": tx.value,
                    "to": tx.to,
                    "from": tx["from"],
                    "input": tx.input,
                    "gasPrice": tx.gasPrice,
                }
            return {"is_profitable": False}
        except Exception as e:
            logger.error(f"Error analyzing ETH transaction {tx.hash.hex()}: {e}")
            return {"is_profitable": False}

    async def _analyze_token_transaction(self, tx) -> Dict[str, Any]:
        """
        Analyze a token transaction (e.g., transfers, swaps) for profitability.
        """
        try:
            if not self.erc20_abi or not tx.input or len(tx.input) < 10:
                logger.debug("Missing ERC20 ABI or invalid transaction input")
                return {"is_profitable": False}
            function_selector = tx.input[:10]
            selector_no_prefix = function_selector[2:]
            function_name = None
            function_params = {}
            decoded = False

            # Attempt direct signature lookup
            if selector_no_prefix in self.function_signatures:
                try:
                    function_name = self.function_signatures[selector_no_prefix]
                    if len(tx.input) >= 138:
                        params_data = tx.input[10:]
                        if function_name == 'transfer':
                            to_address = '0x' + params_data[:64][-40:]
                            amount = int(params_data[64:128], 16)
                            function_params = {'to': to_address, 'amount': amount}
                            decoded = True
                except Exception as e:
                    logger.debug(f"Error in direct signature lookup: {e}")

            # Fallback to ABI-based decoding if direct lookup failed
            if not decoded:
                try:
                    if not tx.to or not self.erc20_abi:
                        logger.debug("Missing contract address or ABI")
                        return {"is_profitable": False}
                    contract = self.web3.eth.contract(
                        address=self.web3.to_checksum_address(tx.to),
                        abi=self.erc20_abi
                    )
                    try:
                        func_obj, decoded_params = contract.decode_function_input(tx.input)
                        function_name = getattr(func_obj, 'fn_name', None) or getattr(func_obj, 'function_identifier', None)
                        if function_name:
                            function_params = decoded_params
                            decoded = True
                    except Exception as e:
                        logger.debug(f"Could not decode function input: {e}")
                except Exception as e:
                    logger.debug(f"Contract decode error: {e}")

            if decoded and function_name in ('transfer', 'transferFrom', 'swap', 'swapExactTokensForTokens', 'swapTokensForExactTokens'):
                params = await self._extract_transaction_params(tx, function_name, function_params)
                if not params:
                    logger.debug(f"Could not extract valid parameters for {function_name}")
                    return {"is_profitable": False}
                amounts = await self._validate_token_amounts(params)
                if not amounts['valid']:
                    logger.debug(f"Invalid token amounts: {amounts['reason']}")
                    if 'details' in amounts:
                        logger.debug(f"Validation details: {amounts['details']}")
                    return {"is_profitable": False}
                estimated_profit = await self._estimate_profit(tx, function_params)
                if estimated_profit > self.minimum_profit_threshold:
                    return {
                        "is_profitable": True,
                        "profit": estimated_profit,
                        "function_name": function_name,
                        "params": function_params,
                        "tx_hash": tx.hash.hex(),
                        "to": tx.to,
                        "input": tx.input,
                        "value": tx.value,
                        "gasPrice": tx.gasPrice,
                    }
            if not decoded:
                logger.debug(f"Could not decode transaction with selector: {function_selector}")
            return {"is_profitable": False}
        except Exception as e:
            logger.error(f"Error analyzing token transaction {tx.hash.hex()}: {e}")
            return {"is_profitable": False}

    async def _is_profitable_eth_transaction(self, tx) -> bool:
        """
        Determine whether an ETH transaction is profitable.
        """
        try:
            potential_profit = await self._estimate_eth_transaction_profit(tx)
            return potential_profit > self.minimum_profit_threshold
        except Exception as e:
            logger.debug(f"Error estimating ETH transaction profit for {tx.hash.hex()}: {e}")
            return False

    async def _estimate_eth_transaction_profit(self, tx: Any) -> Decimal:
        """
        Estimate the profit of an ETH transaction by subtracting gas cost from the transferred value.
        """
        try:
            gas_price_gwei = await self.safetynet.get_dynamic_gas_price()
            gas_used = tx.gas if tx.gas else await self.web3.eth.estimate_gas(tx)
            gas_cost_eth = Decimal(gas_price_gwei) * Decimal(gas_used) * Decimal("1e-9")
            eth_value = Decimal(self.web3.from_wei(tx.value, "ether"))
            potential_profit = eth_value - gas_cost_eth
            logger.debug(f"Estimated ETH tx profit for {tx.hash.hex()[:8]}...: {potential_profit:.6f} ETH")
            return potential_profit if potential_profit > 0 else Decimal(0)
        except Exception as e:
            logger.error(f"Error estimating ETH transaction profit: {e}", exc_info=True)
            return Decimal(0)

    async def _estimate_profit(self, tx, function_params: Dict[str, Any]) -> Decimal:
        """
        Estimate the profit for a token transaction using gas costs, token amounts, and market data.
        """
        try:
            gas_data = await self._calculate_gas_costs(tx)
            if not gas_data['valid']:
                logger.debug(f"Invalid gas data: {gas_data['reason']}")
                return Decimal(0)
            amounts = await self._validate_token_amounts(function_params)
            if not amounts['valid']:
                logger.debug(f"Invalid token amounts: {amounts['reason']}")
                return Decimal(0)
            token_path = function_params.get('path')
            if not token_path or not isinstance(token_path, list):
                logger.debug("Missing or invalid 'path' in function parameters")
                return Decimal(0)
            market_data = await self._get_market_data(token_path[-1])
            if not market_data['valid']:
                logger.debug(f"Invalid market data: {market_data['reason']}")
                return Decimal(0)
            profit = await self._calculate_final_profit(
                amounts=amounts['data'],
                gas_costs=gas_data['data'],
                market_data=market_data['data']
            )
            self._log_profit_calculation(profit, amounts['data'], gas_data['data'], market_data['data'])
            return Decimal(max(0, profit))
        except Exception as e:
            logger.error(f"Error in profit estimation: {e}")
            return Decimal(0)

    async def _calculate_gas_costs(self, tx: Any) -> Dict[str, Any]:
        """
        Calculate gas costs with a 10% margin.
        """
        try:
            gas_price_wei = Decimal(tx.gasPrice)
            gas_price_gwei = Decimal(self.web3.from_wei(gas_price_wei, "gwei"))
            gas_used = tx.gas if tx.gas else await self.web3.eth.estimate_gas(tx)
            gas_used = Decimal(gas_used)
            gas_with_margin = gas_used * Decimal("1.1")
            gas_cost_eth = (gas_price_gwei * gas_with_margin * Decimal("1e-9")).quantize(Decimal("0.000000001"))
            return {
                'valid': True,
                'data': {
                    'gas_price_gwei': gas_price_gwei,
                    'gas_used': gas_used,
                    'gas_with_margin': gas_with_margin,
                    'gas_cost_eth': gas_cost_eth
                }
            }
        except Exception as e:
            return {'valid': False, 'reason': str(e)}

    async def _validate_token_amounts(self, function_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and convert token amounts from function parameters.
        """
        try:
            input_amount = function_params.get("amountIn",
                function_params.get("value",
                function_params.get("amount",
                function_params.get("_value", 0)
            )))
            output_amount = function_params.get("amountOutMin",
                function_params.get("amountOut",
                function_params.get("amount",
                function_params.get("_amount", 0)
                )
                )
            )
            def parse_amount(amount: Any) -> int:
                if isinstance(amount, str):
                    if amount.startswith("0x"):
                        return int(amount, 16)
                    if amount.isnumeric():
                        return int(amount)
                return int(amount) if amount else 0
            try:
                input_amount_wei = Decimal(str(parse_amount(input_amount)))
                output_amount_wei = Decimal(str(parse_amount(output_amount)))
            except (ValueError, TypeError) as e:
                return {
                    'valid': False, 
                    'reason': f'Amount parsing error: {str(e)}',
                    'details': {
                        'input_raw': input_amount,
                        'output_raw': output_amount
                    }
                }
            if input_amount_wei <= 0 and output_amount_wei <= 0:
                return {
                    'valid': False,
                    'reason': 'Both input and output amounts are zero or negative',
                    'details': {
                        'input_wei': str(input_amount_wei),
                        'output_wei': str(output_amount_wei)
                    }
                }
            input_amount_eth = Decimal(self.web3.from_wei(input_amount_wei, "ether")).quantize(Decimal("0.000000001"))
            output_amount_eth = Decimal(self.web3.from_wei(output_amount_wei, "ether")).quantize(Decimal("0.000000001"))
            return {
                'valid': True,
                'data': {
                    'input_eth': input_amount_eth,
                    'output_eth': output_amount_eth,
                    'input_wei': input_amount_wei,
                    'output_wei': output_amount_wei
                }
            }
        except Exception as e:
            logger.error(f"Unexpected error in token amount validation: {e}")
            return {
                'valid': False,
                'reason': 'Unknown validation error',
                'details': {
                    'error': str(e),
                    'params': str(function_params)
                }
            }

    async def _get_market_data(self, token_address: str) -> Dict[str, Any]:
        """
        Retrieve market data for a token given its address.
        """
        try:
            token_symbol = await self.apiconfig.get_token_symbol(token_address)
            if not token_symbol:
                logger.debug(f"Cannot determine token symbol for address {token_address}")
                return {'valid': False, 'reason': 'Token symbol not found'}
            price = await self.apiconfig.get_real_time_price(token_symbol)
            if not price or price <= 0:
                return {'valid': False, 'reason': 'Invalid market price'}
            slippage = await self._calculate_dynamic_slippage(token_symbol)
            return {
                'valid': True,
                'data': {
                    'price': Decimal(str(price)),
                    'slippage': slippage,
                    'symbol': token_symbol
                }
            }
        except Exception as e:
            return {'valid': False, 'reason': str(e)}

    async def _calculate_dynamic_slippage(self, token_symbol: str) -> Decimal:
        """
        Calculate a dynamic slippage factor based on the token's trading volume.
        """
        try:
            volume = await self.apiconfig.get_token_volume(token_symbol)
            if volume > 1_000_000:
                return Decimal("0.995")
            elif volume > 500_000:
                return Decimal("0.99")
            else:
                return Decimal("0.98")
        except Exception:
            return Decimal("0.99")

    async def _calculate_final_profit(
        self,
        amounts: Dict[str, Decimal],
        gas_costs: Dict[str, Decimal],
        market_data: Dict[str, Any]
    ) -> Decimal:
        """
        Calculate the final profit for a token transaction.
        """
        try:
            expected_output_value = (
                amounts['output_eth'] *
                market_data['price'] *
                market_data['slippage']
            ).quantize(Decimal("0.000000001"))
            profit = (
                expected_output_value -
                amounts['input_eth'] -
                gas_costs['gas_cost_eth']
            ).quantize(Decimal("0.000000001"))
            return profit
        except Exception as e:
            logger.error(f"Error in final profit calculation: {e}")
            return Decimal(0)

    async def _log_profit_calculation(
        self,
        profit: Decimal,
        amounts: Dict[str, Decimal],
        gas_costs: Dict[str, Decimal],
        market_data: Dict[str, Any]
    ) -> None:
        """
        Log detailed profit calculation information.
        """
        logger.debug(
            f"Profit Calculation Details:\n"
            f"  Token: {market_data['symbol']}\n"
            f"  Input Amount: {amounts['input_eth']:.9f} ETH\n"
            f"  Expected Output: {amounts['output_eth']:.9f} tokens\n"
            f"  Market Price: {market_data['price']:.9f}\n"
            f"  Slippage: {(1 - float(market_data['slippage'])) * 100:.2f}%\n"
            f"  Gas Cost: {gas_costs['gas_cost_eth']:.9f} ETH\n"
            f"  Gas Price: {gas_costs['gas_price_gwei']:.2f} Gwei\n"
            f"  Final Profit: {profit:.9f} ETH"
        )

    async def _log_transaction_details(self, tx, is_eth=False) -> None:
        """
        Log basic details of a transaction.
        """
        try:
            transaction_info = {
                "transaction hash": tx.hash.hex(),
                "value": self.web3.from_wei(tx.value, "ether") if is_eth else tx.value,
                "from": tx["from"],
                "to": (tx.to[:10] + "..." + tx.to[-10:]) if tx.to else None,
                "input": tx.input,
                "gas price": self.web3.from_wei(tx.gasPrice, "gwei"),
            }
            if is_eth:
                logger.debug(f"Pending ETH Transaction Details: {transaction_info}")
            else:
                logger.debug(f"Pending Token Transaction Details: {transaction_info}")
        except Exception as e:
            logger.debug(f"Error logging transaction details for {tx.hash.hex()}: {e}", exc_info=True)

    async def _extract_transaction_params(
        self,
        tx: Any,
        function_name: str,
        decoded_params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Extract and standardize parameters from a decoded transaction.
        """
        try:
            params = {}
            if function_name in ['transfer', 'transferFrom']:
                amount = decoded_params.get('amount',
                            decoded_params.get('_value',
                                decoded_params.get('value',
                                    decoded_params.get('wad', 0))))
                to_addr = decoded_params.get('to',
                            decoded_params.get('_to',
                                decoded_params.get('dst',
                                    decoded_params.get('recipient'))))
                if function_name == 'transferFrom':
                    from_addr = decoded_params.get('from',
                            decoded_params.get('_from',
                                decoded_params.get('src',
                                    decoded_params.get('sender'))))
                    params['from'] = from_addr
                params.update({'amount': amount, 'to': to_addr})
            elif function_name in ['swap', 'swapExactTokensForTokens', 'swapTokensForExactTokens']:
                params = {
                    'amountIn': decoded_params.get('amountIn',
                                decoded_params.get('amount0',
                                    decoded_params.get('amountInMax', 0))),
                    'amountOutMin': decoded_params.get('amountOutMin',
                                decoded_params.get('amountOut',
                                    decoded_params.get('amount', 0))),
                    'path': decoded_params.get('path', [])
                }
            if not self._validate_params_format(params, function_name):
                logger.debug(f"Invalid parameter format for {function_name}")
                return None
            return params
        except Exception as e:
            logger.error(f"Error extracting transaction parameters: {e}")
            return None

    def _validate_params_format(self, params: Dict[str, Any], function_name: str) -> bool:
        """
        Validate the format of extracted transaction parameters.
        """
        try:
            if function_name in ['transfer', 'transferFrom']:
                required = ['amount', 'to']
                if function_name == 'transferFrom':
                    required.append('from')
            elif function_name in ['swap', 'swapExactTokensForTokens', 'swapTokensForExactTokens']:
                required = ['amountIn', 'amountOutMin', 'path']
            else:
                logger.debug(f"Unsupported function name: {function_name}")
                return False
            for field in required:
                if params.get(field) is None or params.get(field) == '':
                    logger.debug(f"Missing or empty field '{field}' for function '{function_name}'")
                    return False
            return True
        except Exception as e:
            logger.error(f"Parameter validation error: {e}")
            return False

    async def _is_profitable_eth_transaction(self, tx) -> bool:
        """
        Determine whether an ETH transaction is profitable.
        """
        try:
            potential_profit = await self._estimate_eth_transaction_profit(tx)
            return potential_profit > self.minimum_profit_threshold
        except Exception as e:
            logger.debug(f"Error estimating ETH transaction profit for {tx.hash.hex()}: {e}")
            return False

    async def _estimate_eth_transaction_profit(self, tx: Any) -> Decimal:
        """
        Estimate the profit of an ETH transaction by subtracting gas cost from the transferred value.
        """
        try:
            gas_price_gwei = await self.safetynet.get_dynamic_gas_price()
            gas_used = tx.gas if tx.gas else await self.web3.eth.estimate_gas(tx)
            gas_cost_eth = Decimal(gas_price_gwei) * Decimal(gas_used) * Decimal("1e-9")
            eth_value = Decimal(self.web3.from_wei(tx.value, "ether"))
            potential_profit = eth_value - gas_cost_eth
            logger.debug(f"Estimated ETH tx profit for {tx.hash.hex()[:8]}...: {potential_profit:.6f} ETH")
            return potential_profit if potential_profit > 0 else Decimal(0)
        except Exception as e:
            logger.error(f"Error estimating ETH transaction profit: {e}", exc_info=True)
            return Decimal(0)

    async def _estimate_profit(self, tx, function_params: Dict[str, Any]) -> Decimal:
        """
        Estimate the profit for a token transaction using gas costs, token amounts, and market data.
        """
        try:
            gas_data = await self._calculate_gas_costs(tx)
            if not gas_data['valid']:
                logger.debug(f"Invalid gas data: {gas_data['reason']}")
                return Decimal(0)
            amounts = await self._validate_token_amounts(function_params)
            if not amounts['valid']:
                logger.debug(f"Invalid token amounts: {amounts['reason']}")
                return Decimal(0)
            token_path = function_params.get('path')
            if not token_path or not isinstance(token_path, list):
                logger.debug("Missing or invalid 'path' in function parameters")
                return Decimal(0)
            market_data = await self._get_market_data(token_path[-1])
            if not market_data['valid']:
                logger.debug(f"Invalid market data: {market_data['reason']}")
                return Decimal(0)
            profit = await self._calculate_final_profit(
                amounts=amounts['data'],
                gas_costs=gas_data['data'],
                market_data=market_data['data']
            )
            self._log_profit_calculation(profit, amounts['data'], gas_data['data'], market_data['data'])
            return Decimal(max(0, profit))
        except Exception as e:
            logger.error(f"Error in profit estimation: {e}")
            return Decimal(0)

    async def _calculate_gas_costs(self, tx: Any) -> Dict[str, Any]:
        """
        Calculate gas costs with a 10% margin.
        """
        try:
            gas_price_wei = Decimal(tx.gasPrice)
            gas_price_gwei = Decimal(self.web3.from_wei(gas_price_wei, "gwei"))
            gas_used = tx.gas if tx.gas else await self.web3.eth.estimate_gas(tx)
            gas_used = Decimal(gas_used)
            gas_with_margin = gas_used * Decimal("1.1")
            gas_cost_eth = (gas_price_gwei * gas_with_margin * Decimal("1e-9")).quantize(Decimal("0.000000001"))
            return {
                'valid': True,
                'data': {
                    'gas_price_gwei': gas_price_gwei,
                    'gas_used': gas_used,
                    'gas_with_margin': gas_with_margin,
                    'gas_cost_eth': gas_cost_eth
                }
            }
        except Exception as e:
            return {'valid': False, 'reason': str(e)}

    async def _validate_token_amounts(self, function_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and convert token amounts from function parameters.
        """
        try:
            input_amount = function_params.get("amountIn",
                function_params.get("value",
                function_params.get("amount",
                function_params.get("_value", 0)
            )))
            output_amount = function_params.get("amountOutMin",
                function_params.get("amountOut",
                function_params.get("amount",
                function_params.get("_amount", 0)
                )
                )
            )
            def parse_amount(amount: Any) -> int:
                if isinstance(amount, str):
                    if amount.startswith("0x"):
                        return int(amount, 16)
                    if amount.isnumeric():
                        return int(amount)
                return int(amount) if amount else 0
            try:
                input_amount_wei = Decimal(str(parse_amount(input_amount)))
                output_amount_wei = Decimal(str(parse_amount(output_amount)))
            except (ValueError, TypeError) as e:
                return {
                    'valid': False, 
                    'reason': f'Amount parsing error: {str(e)}',
                    'details': {
                        'input_raw': input_amount,
                        'output_raw': output_amount
                    }
                }
            if input_amount_wei <= 0 and output_amount_wei <= 0:
                return {
                    'valid': False,
                    'reason': 'Both input and output amounts are zero or negative',
                    'details': {
                        'input_wei': str(input_amount_wei),
                        'output_wei': str(output_amount_wei)
                    }
                }
            input_amount_eth = Decimal(self.web3.from_wei(input_amount_wei, "ether")).quantize(Decimal("0.000000001"))
            output_amount_eth = Decimal(self.web3.from_wei(output_amount_wei, "ether")).quantize(Decimal("0.000000001"))
            return {
                'valid': True,
                'data': {
                    'input_eth': input_amount_eth,
                    'output_eth': output_amount_eth,
                    'input_wei': input_amount_wei,
                    'output_wei': output_amount_wei
                }
            }
        except Exception as e:
            logger.error(f"Unexpected error in token amount validation: {e}")
            return {
                'valid': False,
                'reason': 'Unknown validation error',
                'details': {
                    'error': str(e),
                    'params': str(function_params)
                }
            }

    async def _get_market_data(self, token_address: str) -> Dict[str, Any]:
        """
        Retrieve market data for a token given its address.
        """
        try:
            token_symbol = await self.apiconfig.get_token_symbol(token_address)
            if not token_symbol:
                logger.debug(f"Cannot get token symbol for address {token_address}")
                return {'valid': False, 'reason': 'Token symbol not found'}
            price = await self.apiconfig.get_real_time_price(token_symbol)
            if not price or price <= 0:
                return {'valid': False, 'reason': 'Invalid market price'}
            slippage = await self._calculate_dynamic_slippage(token_symbol)
            return {
                'valid': True,
                'data': {
                    'price': Decimal(str(price)),
                    'slippage': slippage,
                    'symbol': token_symbol
                }
            }
        except Exception as e:
            return {'valid': False, 'reason': str(e)}

    async def _calculate_dynamic_slippage(self, token_symbol: str) -> Decimal:
        """
        Calculate a dynamic slippage factor based on the token's trading volume.
        """
        try:
            volume = await self.apiconfig.get_token_volume(token_symbol)
            if volume > 1_000_000:
                return Decimal("0.995")
            elif volume > 500_000:
                return Decimal("0.99")
            else:
                return Decimal("0.98")
        except Exception:
            return Decimal("0.99")

    async def _calculate_final_profit(
        self,
        amounts: Dict[str, Decimal],
        gas_costs: Dict[str, Decimal],
        market_data: Dict[str, Any]
    ) -> Decimal:
        """
        Calculate the final profit for a token transaction.
        """
        try:
            expected_output_value = (
                amounts['output_eth'] *
                market_data['price'] *
                market_data['slippage']
            ).quantize(Decimal("0.000000001"))
            profit = (
                expected_output_value -
                amounts['input_eth'] -
                gas_costs['gas_cost_eth']
            ).quantize(Decimal("0.000000001"))
            return profit
        except Exception as e:
            logger.error(f"Error in final profit calculation: {e}")
            return Decimal(0)

    async def _log_profit_calculation(
        self,
        profit: Decimal,
        amounts: Dict[str, Decimal],
        gas_costs: Dict[str, Decimal],
        market_data: Dict[str, Any]
    ) -> None:
        """
        Log detailed information about the profit calculation for debugging purposes.
        """
        profitable = "Yes" if profit > Decimal("0") else "No"
        logger.debug(
            f"Profit Calculation Summary:\n"
            f"  Token: {market_data['symbol']}\n"
            f"  Input Amount: {amounts['input_eth']:.9f} ETH\n"
            f"  Expected Output: {amounts['output_eth']:.9f} tokens\n"
            f"  Market Price: {market_data['price']:.9f}\n"
            f"  Slippage: {(1 - float(market_data['slippage'])) * 100:.2f}%\n"
            f"  Gas Cost: {gas_costs['gas_cost_eth']:.9f} ETH\n"
            f"  Gas Price: {gas_costs['gas_price_gwei']:.2f} Gwei\n"
            f"  Final Profit: {profit:.9f} ETH\n"
            f"  Profitable: {profitable}"
        )

    async def _log_transaction_details(self, tx, is_eth=False) -> None:
        """
        Log basic details of a transaction.
        """
        try:
            transaction_info = {
                "transaction hash": tx.hash.hex(),
                "value": self.web3.from_wei(tx.value, "ether") if is_eth else tx.value,
                "from": tx["from"],
                "to": (tx.to[:10] + "..." + tx.to[-10:]) if tx.to else None,
                "input": tx.input,
                "gas price": self.web3.from_wei(tx.gasPrice, "gwei"),
            }
            if is_eth:
                logger.debug(f"Pending ETH Transaction Details: {transaction_info}")
            else:
                logger.debug(f"Pending Token Transaction Details: {transaction_info}")
        except Exception as e:
            logger.debug(f"Error logging transaction details for {tx.hash.hex()}: {e}", exc_info=True)

    async def _extract_transaction_params(
        self,
        tx: Any,
        function_name: str,
        decoded_params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Extract and standardize parameters from a decoded transaction.
        """
        try:
            params = {}
            if function_name in ['transfer', 'transferFrom']:
                amount = decoded_params.get('amount',
                            decoded_params.get('_value',
                                decoded_params.get('value',
                                    decoded_params.get('wad', 0))))
                to_addr = decoded_params.get('to',
                            decoded_params.get('_to',
                                decoded_params.get('dst',
                                    decoded_params.get('recipient'))))
                if function_name == 'transferFrom':
                    from_addr = decoded_params.get('from',
                            decoded_params.get('_from',
                                decoded_params.get('src',
                                    decoded_params.get('sender'))))
                    params['from'] = from_addr
                params.update({'amount': amount, 'to': to_addr})
            elif function_name in ['swap', 'swapExactTokensForTokens', 'swapTokensForExactTokens']:
                params = {
                    'amountIn': decoded_params.get('amountIn',
                                decoded_params.get('amount0',
                                    decoded_params.get('amountInMax', 0))),
                    'amountOutMin': decoded_params.get('amountOutMin',
                                decoded_params.get('amountOut',
                                    decoded_params.get('amount', 0))),
                    'path': decoded_params.get('path', [])
                }
            if not self._validate_params_format(params, function_name):
                logger.debug(f"Invalid parameter format for {function_name}")
                return None
            return params
        except Exception as e:
            logger.error(f"Error extracting transaction parameters: {e}")
            return None

    def _validate_params_format(self, params: Dict[str, Any], function_name: str) -> bool:
        """
        Validate the format of extracted transaction parameters.
        """
        try:
            if function_name in ['transfer', 'transferFrom']:
                required = ['amount', 'to']
                if function_name == 'transferFrom':
                    required.append('from')
            elif function_name in ['swap', 'swapExactTokensForTokens', 'swapTokensForExactTokens']:
                required = ['amountIn', 'amountOutMin', 'path']
            else:
                logger.debug(f"Unsupported function name: {function_name}")
                return False
            for field in required:
                if params.get(field) is None or params.get(field) == '':
                    logger.debug(f"Missing or empty field '{field}' for function '{function_name}'")
                    return False
            return True
        except Exception as e:
            logger.error(f"Parameter validation error: {e}")
            return False

    async def _is_profitable_eth_transaction(self, tx) -> bool:
        """
        Determine whether an ETH transaction is profitable.
        """
        try:
            potential_profit = await self._estimate_eth_transaction_profit(tx)
            return potential_profit > self.minimum_profit_threshold
        except Exception as e:
            logger.debug(f"Error estimating ETH transaction profit for {tx.hash.hex()}: {e}")
            return False

    async def _estimate_eth_transaction_profit(self, tx: Any) -> Decimal:
        """
        Estimate the profit of an ETH transaction by subtracting gas cost from the transferred value.
        """
        try:
            gas_price_gwei = await self.safetynet.get_dynamic_gas_price()
            gas_used = tx.gas if tx.gas else await self.web3.eth.estimate_gas(tx)
            gas_cost_eth = Decimal(gas_price_gwei) * Decimal(gas_used) * Decimal("1e-9")
            eth_value = Decimal(self.web3.from_wei(tx.value, "ether"))
            potential_profit = eth_value - gas_cost_eth
            logger.debug(f"Estimated ETH tx profit for {tx.hash.hex()[:8]}...: {potential_profit:.6f} ETH")
            return potential_profit if potential_profit > 0 else Decimal(0)
        except Exception as e:
            logger.error(f"Error estimating ETH transaction profit: {e}", exc_info=True)
            return Decimal(0)

    async def _estimate_profit(self, tx, function_params: Dict[str, Any]) -> Decimal:
        """
        Estimate the profit for a token transaction using gas costs, token amounts, and market data.
        """
        try:
            gas_data = await self._calculate_gas_costs(tx)
            if not gas_data['valid']:
                logger.debug(f"Invalid gas data: {gas_data['reason']}")
                return Decimal(0)
            amounts = await self._validate_token_amounts(function_params)
            if not amounts['valid']:
                logger.debug(f"Invalid token amounts: {amounts['reason']}")
                return Decimal(0)
            token_path = function_params.get('path')
            if not token_path or not isinstance(token_path, list):
                logger.debug("Missing or invalid 'path' in function parameters")
                return Decimal(0)
            market_data = await self._get_market_data(token_path[-1])
            if not market_data['valid']:
                logger.debug(f"Invalid market data: {market_data['reason']}")
                return Decimal(0)
            profit = await self._calculate_final_profit(
                amounts=amounts['data'],
                gas_costs=gas_data['data'],
                market_data=market_data['data']
            )
            self._log_profit_calculation(profit, amounts['data'], gas_data['data'], market_data['data'])
            return Decimal(max(0, profit))
        except Exception as e:
            logger.error(f"Error in profit estimation: {e}")
            return Decimal(0)

    async def _calculate_gas_costs(self, tx: Any) -> Dict[str, Any]:
        """
        Calculate gas costs with a 10% margin.
        """
        try:
            gas_price_wei = Decimal(tx.gasPrice)
            gas_price_gwei = Decimal(self.web3.from_wei(gas_price_wei, "gwei"))
            gas_used = tx.gas if tx.gas else await self.web3.eth.estimate_gas(tx)
            gas_used = Decimal(gas_used)
            gas_with_margin = gas_used * Decimal("1.1")
            gas_cost_eth = (gas_price_gwei * gas_with_margin * Decimal("1e-9")).quantize(Decimal("0.000000001"))
            return {
                'valid': True,
                'data': {
                    'gas_price_gwei': gas_price_gwei,
                    'gas_used': gas_used,
                    'gas_with_margin': gas_with_margin,
                    'gas_cost_eth': gas_cost_eth
                }
            }
        except Exception as e:
            return {'valid': False, 'reason': str(e)}

    async def _validate_token_amounts(self, function_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and convert token amounts from function parameters.
        """
        try:
            input_amount = function_params.get("amountIn",
                function_params.get("value",
                function_params.get("amount",
                function_params.get("_value", 0)
            )))
            output_amount = function_params.get("amountOutMin",
                function_params.get("amountOut",
                function_params.get("amount",
                function_params.get("_amount", 0)
                )
                )
            )
            def parse_amount(amount: Any) -> int:
                if isinstance(amount, str):
                    if amount.startswith("0x"):
                        return int(amount, 16)
                    if amount.isnumeric():
                        return int(amount)
                return int(amount) if amount else 0
            try:
                input_amount_wei = Decimal(str(parse_amount(input_amount)))
                output_amount_wei = Decimal(str(parse_amount(output_amount)))
            except (ValueError, TypeError) as e:
                return {
                    'valid': False,
                    'reason': f'Amount parsing error: {str(e)}',
                    'details': {
                        'input_raw': input_amount,
                        'output_raw': output_amount
                    }
                }
            if input_amount_wei <= 0 and output_amount_wei <= 0:
                return {
                    'valid': False,
                    'reason': 'Both input and output amounts are zero or negative',
                    'details': {
                        'input_wei': str(input_amount_wei),
                        'output_wei': str(output_amount_wei)
                    }
                }
            input_amount_eth = Decimal(self.web3.from_wei(input_amount_wei, "ether")).quantize(Decimal("0.000000001"))
            output_amount_eth = Decimal(self.web3.from_wei(output_amount_wei, "ether")).quantize(Decimal("0.000000001"))
            return {
                'valid': True,
                'data': {
                    'input_eth': input_amount_eth,
                    'output_eth': output_amount_eth,
                    'input_wei': input_amount_wei,
                    'output_wei': output_amount_wei
                }
            }
        except Exception as e:
            logger.error(f"Unexpected error in token amount validation: {e}")
            return {
                'valid': False,
                'reason': 'Unknown validation error',
                'details': {
                    'error': str(e),
                    'params': str(function_params)
                }
            }

    async def _get_market_data(self, token_address: str) -> Dict[str, Any]:
        """
        Retrieve market data for a token given its address.
        """
        try:
            token_symbol = await self.apiconfig.get_token_symbol(token_address)
            if not token_symbol:
                logger.debug(f"Cannot get token symbol for address {token_address}")
                return {'valid': False, 'reason': 'Token symbol not found'}
            price = await self.apiconfig.get_real_time_price(token_symbol)
            if not price or price <= 0:
                return {'valid': False, 'reason': 'Invalid market price'}
            slippage = await self._calculate_dynamic_slippage(token_symbol)
            return {
                'valid': True,
                'data': {
                    'price': Decimal(str(price)),
                    'slippage': slippage,
                    'symbol': token_symbol
                }
            }
        except Exception as e:
            return {'valid': False, 'reason': str(e)}

    async def _calculate_dynamic_slippage(self, token_symbol: str) -> Decimal:
        """
        Calculate a dynamic slippage factor based on the token's trading volume.
        """
        try:
            volume = await self.apiconfig.get_token_volume(token_symbol)
            if volume > 1_000_000:
                return Decimal("0.995")
            elif volume > 500_000:
                return Decimal("0.99")
            else:
                return Decimal("0.98")
        except Exception:
            return Decimal("0.99")

    async def _calculate_final_profit(
        self,
        amounts: Dict[str, Decimal],
        gas_costs: Dict[str, Decimal],
        market_data: Dict[str, Any]
    ) -> Decimal:
        """
        Calculate the final profit for a token transaction.
        """
        try:
            expected_output_value = (
                amounts['output_eth'] *
                market_data['price'] *
                market_data['slippage']
            ).quantize(Decimal("0.000000001"))
            profit = (
                expected_output_value -
                amounts['input_eth'] -
                gas_costs['gas_cost_eth']
            ).quantize(Decimal("0.000000001"))
            return profit
        except Exception as e:
            logger.error(f"Error in final profit calculation: {e}")
            return Decimal(0)

    async def _log_profit_calculation(
        self,
        profit: Decimal,
        amounts: Dict[str, Decimal],
        gas_costs: Dict[str, Decimal],
        market_data: Dict[str, Any]
    ) -> None:
        """
        Log detailed profit calculation information.
        """
        logger.debug(
            f"Profit Calculation Details:\n"
            f"  Token: {market_data['symbol']}\n"
            f"  Input Amount: {amounts['input_eth']:.9f} ETH\n"
            f"  Expected Output: {amounts['output_eth']:.9f} tokens\n"
            f"  Market Price: {market_data['price']:.9f}\n"
            f"  Slippage: {(1 - float(market_data['slippage'])) * 100:.2f}%\n"
            f"  Gas Cost: {gas_costs['gas_cost_eth']:.9f} ETH\n"
            f"  Gas Price: {gas_costs['gas_price_gwei']:.2f} Gwei\n"
            f"  Final Profit: {profit:.9f} ETH"
        )

    async def _log_transaction_details(self, tx, is_eth=False) -> None:
        """
        Log basic details of a transaction.
        """
        try:
            transaction_info = {
                "transaction hash": tx.hash.hex(),
                "value": self.web3.from_wei(tx.value, "ether") if is_eth else tx.value,
                "from": tx["from"],
                "to": (tx.to[:10] + "..." + tx.to[-10:]) if tx.to else None,
                "input": tx.input,
                "gas price": self.web3.from_wei(tx.gasPrice, "gwei"),
            }
            if is_eth:
                logger.debug(f"Pending ETH Transaction Details: {transaction_info}")
            else:
                logger.debug(f"Pending Token Transaction Details: {transaction_info}")
        except Exception as e:
            logger.debug(f"Error logging transaction details for {tx.hash.hex()}: {e}", exc_info=True)

    async def _extract_transaction_params(
        self,
        tx: Any,
        function_name: str,
        decoded_params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Extract and standardize parameters from a decoded transaction.
        """
        try:
            params = {}
            if function_name in ['transfer', 'transferFrom']:
                amount = decoded_params.get('amount',
                            decoded_params.get('_value',
                                decoded_params.get('value',
                                    decoded_params.get('wad', 0))))
                to_addr = decoded_params.get('to',
                            decoded_params.get('_to',
                                decoded_params.get('dst',
                                    decoded_params.get('recipient'))))
                if function_name == 'transferFrom':
                    from_addr = decoded_params.get('from',
                            decoded_params.get('_from',
                                decoded_params.get('src',
                                    decoded_params.get('sender'))))
                    params['from'] = from_addr
                params.update({'amount': amount, 'to': to_addr})
            elif function_name in ['swap', 'swapExactTokensForTokens', 'swapTokensForExactTokens']:
                params = {
                    'amountIn': decoded_params.get('amountIn',
                                decoded_params.get('amount0',
                                    decoded_params.get('amountInMax', 0))),
                    'amountOutMin': decoded_params.get('amountOutMin',
                                decoded_params.get('amountOut',
                                    decoded_params.get('amount', 0))),
                    'path': decoded_params.get('path', [])
                }
            if not self._validate_params_format(params, function_name):
                logger.debug(f"Invalid parameter format for {function_name}")
                return None
            return params
        except Exception as e:
            logger.error(f"Error extracting transaction parameters: {e}")
            return None

    def _validate_params_format(self, params: Dict[str, Any], function_name: str) -> bool:
        """
        Validate the format of extracted transaction parameters.
        """
        try:
            if function_name in ['transfer', 'transferFrom']:
                required = ['amount', 'to']
                if function_name == 'transferFrom':
                    required.append('from')
            elif function_name in ['swap', 'swapExactTokensForTokens', 'swapTokensForExactTokens']:
                required = ['amountIn', 'amountOutMin', 'path']
            else:
                logger.debug(f"Unsupported function name: {function_name}")
                return False
            for field in required:
                if params.get(field) is None or params.get(field) == '':
                    logger.debug(f"Missing or empty field '{field}' for function '{function_name}'")
                    return False
            return True
        except Exception as e:
            logger.error(f"Parameter validation error: {e}")
            return False

    async def _is_profitable_eth_transaction(self, tx) -> bool:
        """
        Determine whether an ETH transaction is profitable.
        """
        try:
            potential_profit = await self._estimate_eth_transaction_profit(tx)
            return potential_profit > self.minimum_profit_threshold
        except Exception as e:
            logger.debug(f"Error estimating ETH transaction profit for {tx.hash.hex()}: {e}")
            return False

    async def _estimate_eth_transaction_profit(self, tx: Any) -> Decimal:
        """
        Estimate the profit of an ETH transaction by subtracting gas cost from the transferred value.
        """
        try:
            gas_price_gwei = await self.safetynet.get_dynamic_gas_price()
            gas_used = tx.gas if tx.gas else await self.web3.eth.estimate_gas(tx)
            gas_cost_eth = Decimal(gas_price_gwei) * Decimal(gas_used) * Decimal("1e-9")
            eth_value = Decimal(self.web3.from_wei(tx.value, "ether"))
            potential_profit = eth_value - gas_cost_eth
            logger.debug(f"Estimated ETH tx profit for {tx.hash.hex()[:8]}...: {potential_profit:.6f} ETH")
            return potential_profit if potential_profit > 0 else Decimal(0)
        except Exception as e:
            logger.error(f"Error estimating ETH transaction profit: {e}", exc_info=True)
            return Decimal(0)

    async def _estimate_profit(self, tx, function_params: Dict[str, Any]) -> Decimal:
        """
        Estimate the profit for a token transaction using gas costs, token amounts, and market data.
        """
        try:
            gas_data = await self._calculate_gas_costs(tx)
            if not gas_data['valid']:
                logger.debug(f"Invalid gas data: {gas_data['reason']}")
                return Decimal(0)
            amounts = await self._validate_token_amounts(function_params)
            if not amounts['valid']:
                logger.debug(f"Invalid token amounts: {amounts['reason']}")
                return Decimal(0)
            token_path = function_params.get('path')
            if not token_path or not isinstance(token_path, list):
                logger.debug("Missing or invalid 'path' in function parameters")
                return Decimal(0)
            market_data = await self._get_market_data(token_path[-1])
            if not market_data['valid']:
                logger.debug(f"Invalid market data: {market_data['reason']}")
                return Decimal(0)
            profit = await self._calculate_final_profit(
                amounts=amounts['data'],
                gas_costs=gas_data['data'],
                market_data=market_data['data']
            )
            self._log_profit_calculation(profit, amounts['data'], gas_data['data'], market_data['data'])
            return Decimal(max(0, profit))
        except Exception as e:
            logger.error(f"Error in profit estimation: {e}")
            return Decimal(0)

    async def _get_transaction_with_retry(self, tx_hash: str) -> Optional[Any]:
        """
        Retrieve a transaction with retries and exponential backoff.
        """
        backoff = self.configuration.MEMPOOL_RETRY_DELAY
        for attempt in range(self.configuration.MEMPOOL_MAX_RETRIES):
            try:
                if tx_hash in self.cache:
                    return self.cache[tx_hash]
                tx = await self.web3.eth.get_transaction(tx_hash)
                self.cache[tx_hash] = tx
                return tx
            except TransactionNotFound:
                if attempt == self.configuration.MEMPOOL_MAX_RETRIES - 1:
                    return None
                await asyncio.sleep(backoff)
                backoff *= self.backoff_factor
            except Exception as e:
                error_str = str(e)
                if "indexing is in progress" in error_str:
                    if attempt < self.configuration.MEMPOOL_MAX_RETRIES - 1:
                        await asyncio.sleep(backoff)
                        backoff *= self.backoff_factor
                        continue
                logger.error(f"Error fetching transaction {tx_hash}: {e}")
                return None

    async def _handle_profitable_transaction(self, analysis: Dict[str, Any]) -> None:
        """
        Handle a profitable transaction by logging details, tagging it with a strategy type,
        and queuing it for further processing.
        """
        try:
            profit = analysis.get('profit', Decimal(0))
            if isinstance(profit, (int, float)):
                profit = Decimal(str(profit))
            elif not isinstance(profit, Decimal):
                logger.warning(f"Invalid profit type: {type(profit)}")
                profit = Decimal(0)
            profit_str = f"{float(profit):.6f}" if profit > 0 else 'Unknown'
            analysis['profit'] = profit
            analysis['timestamp'] = time.time()
            analysis['gas_price'] = self.web3.from_wei(analysis.get('gasPrice', 0), 'gwei')
            analysis['strategy_type'] = self._determine_strategy_type(analysis)
            await self.profitable_transactions.put(analysis)
            logger.debug(
                f"Profitable transaction identified: {analysis['tx_hash']} "
                f"(Estimated profit: {profit_str} ETH, Strategy Type: {analysis['strategy_type']})"
            )
        except Exception as e:
            logger.error(f"Error handling profitable transaction: {e}")

    def _determine_strategy_type(self, analysis: Dict[str, Any]) -> str:
        """
        Determine the strategy type for a transaction based on its analysis.
        """
        if analysis.get('value', 0) > 0 and 'input' not in analysis:
            return "eth_transaction"
        if analysis.get('function_name') in ("swap", "swapExactTokensForTokens", "swapTokensForExactTokens"):
            if "amountOutMin" in analysis.get('params', {}):
                if 'path' in analysis.get('params', {}):
                    return "sandwich_attack"
                return "back_run"
            return "front_run"
        return "unknown"

    async def _calculate_opportunity_score(
        self,
        price_change: float,
        volatility: float,
        market_conditions: Dict[str, bool],
        current_price: float, 
        historical_prices: List[float]
    ) -> float:
        """
        Calculate a comprehensive opportunity score (0-100) for front-run strategies based on multiple metrics.
        """
        score = 0
        components = {
           "price_change": {
               "very_strong": {"threshold": 5.0, "points": 40},
               "strong": {"threshold": 3.0, "points": 30},
               "moderate": {"threshold": 1.0, "points": 20},
               "slight": {"threshold": 0.5, "points": 10}
           },
           "volatility": {
               "very_low": {"threshold": 0.02, "points": 20},
               "low": {"threshold": 0.05, "points": 15},
               "moderate": {"threshold": 0.08, "points": 10},
           },
           "market_conditions": {
                "bullish_trend": {"points": 10},
                "not_high_volatility": {"points": 5},
                "not_low_liquidity": {"points": 5},
           },
            "price_trend": {
                "upward": {"points": 20},
                "stable": {"points": 10},
           }
       }

        if price_change > components["price_change"]["very_strong"]["threshold"]:
            score += components["price_change"]["very_strong"]["points"]
        elif price_change > components["price_change"]["strong"]["threshold"]:
            score += components["price_change"]["strong"]["points"]
        elif price_change > components["price_change"]["moderate"]["threshold"]:
            score += components["price_change"]["moderate"]["points"]
        elif price_change > components["price_change"]["slight"]["threshold"]:
            score += components["price_change"]["slight"]["points"]

        if historical_prices:
            avg_price = sum(historical_prices) / len(historical_prices)
            if current_price > avg_price * 1.1:
                score += 10
            elif current_price > avg_price * 1.05:
                score += 5

        if volatility < components["volatility"]["very_low"]["threshold"]:
           score += components["volatility"]["very_low"]["points"]
        elif volatility < components["volatility"]["low"]["threshold"]:
           score += components["volatility"]["low"]["points"]
        elif volatility < components["volatility"]["moderate"]["threshold"]:
           score += components["volatility"]["moderate"]["points"]

        if market_conditions.get("bullish_trend", False):
            score += components["market_conditions"]["bullish_trend"]["points"]
        if not market_conditions.get("high_volatility", True):
            score += components["market_conditions"]["not_high_volatility"]["points"]
        if not market_conditions.get("low_liquidity", True):
            score += components["market_conditions"]["not_low_liquidity"]["points"]

        if historical_prices and len(historical_prices) > 1:
            recent_trend = (historical_prices[-1] / historical_prices[0] - 1) * 100
            if recent_trend > 0:
                score += components["price_trend"]["upward"]["points"]
            elif recent_trend > -1:
                score += components["price_trend"]["stable"]["points"]

        logger.debug(f"Calculated opportunity score: {score}/100")
        return score

    async def _calculate_volatility_score(
        self,
        historical_prices: List[float],
        current_price: float,
        market_conditions: Dict[str, bool]
    ) -> float:
        """
        Calculate a volatility score (0-100) based on historical price data and market conditions.
        """
        score = 0
        components = {
            "historical_volatility": {
                "very_high": {"threshold": 0.1, "points": 40},
                "high": {"threshold": 0.08, "points": 30},
                "moderate": {"threshold": 0.05, "points": 20},
                "low": {"threshold": 0.03, "points": 10}
            },
            "price_range": {
                "very_wide": {"threshold": 0.2, "points": 30},
                "wide": {"threshold": 0.15, "points": 20},
                "moderate": {"threshold": 0.1, "points": 10}
            },
            "market_conditions": {
                "high_volatility": {"points": 20},
                "low_liquidity": {"points": 10}
            }
        }

        if historical_prices and len(historical_prices) > 1:
            historical_volatility = np.std(historical_prices) / np.mean(historical_prices)
            if historical_volatility > components["historical_volatility"]["very_high"]["threshold"]:
                score += components["historical_volatility"]["very_high"]["points"]
            elif historical_volatility > components["historical_volatility"]["high"]["threshold"]:
                score += components["historical_volatility"]["high"]["points"]
            elif historical_volatility > components["historical_volatility"]["moderate"]["threshold"]:
                score += components["historical_volatility"]["moderate"]["points"]
            elif historical_volatility > components["historical_volatility"]["low"]["threshold"]:
                score += components["historical_volatility"]["low"]["points"]

        if historical_prices:
            price_range = (max(historical_prices) - min(historical_prices)) / np.mean(historical_prices)
            if price_range > components["price_range"]["very_wide"]["threshold"]:
                score += components["price_range"]["very_wide"]["points"]
            elif price_range > components["price_range"]["wide"]["threshold"]:
                score += components["price_range"]["wide"]["points"]
            elif price_range > components["price_range"]["moderate"]["threshold"]:
                score += components["price_range"]["moderate"]["points"]

        if market_conditions.get("high_volatility", False):
            score += components["market_conditions"]["high_volatility"]["points"]
        if market_conditions.get("low_liquidity", False):
            score += components["market_conditions"]["low_liquidity"]["points"]

        logger.debug(f"Calculated volatility score: {score}/100")
        return score

    async def stop(self) -> None:
        """
        Gracefully stop the MempoolMonitor.
        """
        self.running = False
        logger.info("MempoolMonitor stopped.")
# --- End file: mempoolmonitor.py ---
