# safetynet.py
import asyncio
from decimal import Decimal
from typing import Any, Dict, Tuple, Optional
from cachetools import TTLCache
from web3 import AsyncWeb3
from eth_account import Account
from configuration import Configuration
from apiconfig import APIConfig
from loggingconfig import setup_logging

logger = setup_logging("Safety_Net", level="DEBUG")  

class SafetyNet:
    def __init__(self, web3: AsyncWeb3, configuration: Configuration, address: Optional[str] = None, account: Optional[Account] = None, apiconfig: Optional[APIConfig] = None, marketmonitor: Optional[Any] = None):
        self.web3 = web3
        self.configuration = configuration
        self.address = address
        self.account = account
        self.apiconfig = apiconfig
        self.marketmonitor = marketmonitor
        self.price_cache = TTLCache(maxsize=2000, ttl=self.configuration.SAFETYNET_CACHE_TTL)
        self.gas_price_cache = TTLCache(maxsize=1, ttl=self.configuration.SAFETYNET_GAS_PRICE_TTL)

    async def initialize(self) -> None:
        if not self.web3 or not await self.web3.is_connected():
            raise RuntimeError("Web3 not connected")

    async def get_balance(self, account: Account) -> Decimal:
        key = f"balance:{account.address}"
        if key in self.price_cache:
            return self.price_cache[key]
        for i in range(3):
            try:
                b = await self.web3.eth.get_balance(account.address)
                e = Decimal(self.web3.from_wei(b, "ether"))
                self.price_cache[key] = e
                return e
            except Exception:
                await asyncio.sleep(2 ** i)
        return Decimal("0")

    async def ensure_profit(self, tx: Dict[str, Any], minimum_profit_eth: Optional[float] = None) -> bool:
        minimum = Decimal(minimum_profit_eth or self.configuration.MIN_PROFIT)
        rt = await self.apiconfig.get_real_time_price(tx["output_token"])
        if rt is None:
            return False
        gas_price = Decimal(tx["gas_price"])
        gas_used = Decimal(tx["gas_used"])
        gas_cost = self._calculate_gas_cost(gas_price, gas_used)
        slippage = await self.adjust_slippage_tolerance()
        profit = await self._calculate_profit(tx, rt, slippage, gas_cost)
        return profit > minimum

    def _validate_gas_parameters(self, gas_price_gwei: Decimal, gas_used: Decimal) -> bool:
        if gas_used <= 0:
            return False
        if gas_price_gwei > Decimal(self.configuration.MAX_GAS_PRICE_GWEI):
            return False
        return True

    def _calculate_gas_cost(self, gas_price_gwei: Decimal, gas_used: Decimal) -> Decimal:
        return (gas_price_gwei * gas_used * Decimal("1e-9")).quantize(Decimal("0.000000001"))

    async def _calculate_profit(self, tx: Dict[str, Any], real_price: Decimal, slippage: float, gas_cost: Decimal) -> Decimal:
        expected = real_price * Decimal(tx["amountOut"])
        input_amount = Decimal(tx["amountIn"])
        adjusted = expected * (Decimal("1") - Decimal(slippage))
        profit = adjusted - input_amount - gas_cost
        return profit.quantize(Decimal("0.000000001"))

    async def estimate_gas(self, tx: Dict[str, Any]) -> int:
        try:
            return await self.web3.eth.estimate_gas(tx)
        except Exception:
            return self.configuration.BASE_GAS_LIMIT

    async def adjust_slippage_tolerance(self) -> float:
        congestion = await self.get_network_congestion()
        default = self.configuration.SLIPPAGE_DEFAULT
        if congestion > 0.8:
            s = self.configuration.SLIPPAGE_HIGH_CONGESTION
        elif congestion < 0.2:
            s = self.configuration.SLIPPAGE_LOW_CONGESTION
        else:
            s = default
        return max(self.configuration.MIN_SLIPPAGE, min(s, self.configuration.MAX_SLIPPAGE))

    async def get_network_congestion(self) -> float:
        try:
            b = await self.web3.eth.get_block("latest")
            used = Decimal(b["gasUsed"])
            limit = Decimal(b["gasLimit"])
            return float((used / limit).quantize(Decimal("0.0001")))
        except Exception:
            return 0.5

    async def check_transaction_safety(self, tx: Dict[str, Any], check_type: str = "all") -> Tuple[bool, Dict[str, Any]]:
        messages = []
        gas_ok = True
        profit_ok = True
        congestion_ok = True
        if check_type in ("all", "gas"):
            current = await self.get_dynamic_gas_price()
            if current > Decimal(self.configuration.MAX_GAS_PRICE_GWEI):
                gas_ok = False
                messages.append(f"Gas {current} exceeds {self.configuration.MAX_GAS_PRICE_GWEI}")
        if check_type in ("all", "profit"):
            rt = await self.apiconfig.get_real_time_price(tx["output_token"])
            gas_used = Decimal(tx["gas_used"])
            gas_price = Decimal(tx["gas_price"])
            gas_cost = self._calculate_gas_cost(gas_price, gas_used)
            slippage = await self.adjust_slippage_tolerance()
            profit = await self._calculate_profit(tx, rt or Decimal("0"), slippage, gas_cost)
            if profit < Decimal(self.configuration.MIN_PROFIT):
                profit_ok = False
                messages.append(f"Profit {profit} below {self.configuration.MIN_PROFIT}")
        if check_type in ("all", "network"):
            congestion = await self.get_network_congestion()
            if congestion > 0.8:
                congestion_ok = False
                messages.append(f"Congestion {congestion}")
        is_safe = gas_ok and profit_ok and congestion_ok
        return is_safe, {"gas_ok": gas_ok, "profit_ok": profit_ok, "congestion_ok": congestion_ok, "messages": messages}

    async def get_dynamic_gas_price(self) -> Decimal:
        if "gas_price" in self.gas_price_cache:
            return self.gas_price_cache["gas_price"]
        try:
            b = await self.web3.eth.get_block("latest")
            base_fee = b.get("baseFeePerGas")
            if base_fee:
                price = Decimal(str(self.web3.from_wei(base_fee * 2, "gwei")))
            else:
                raw = await self.web3.eth.gas_price
                price = Decimal(str(self.web3.from_wei(raw, "gwei")))
            self.gas_price_cache["gas_price"] = price
            return price
        except Exception:
            return Decimal(self.configuration.MAX_GAS_PRICE_GWEI)

    async def validate_transaction(self, tx: Dict[str, Any]) -> bool:
        try:
            for field in ("output_token", "amountOut", "amountIn", "gas_price", "gas_used"):
                if field not in tx:
                    return False
            return self._validate_gas_parameters(Decimal(tx["gas_price"]), Decimal(tx["gas_used"]))
        except Exception:
            return False

    async def stop(self) -> None:
        await asyncio.sleep(0)
