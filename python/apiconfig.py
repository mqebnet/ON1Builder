import asyncio
import os
import aiohttp
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from cachetools import TTLCache
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

from configuration import Configuration
from loggingconfig import setup_logging

logger = setup_logging("API_Config", level="DEBUG")  

class APIConfig:
    MAX_REQUEST_ATTEMPTS = 5
    REQUEST_BACKOFF_FACTOR = 1.5

    def __init__(self, configuration: Configuration) -> None:
        self.configuration = configuration
        self.session: Optional[aiohttp.ClientSession] = None
        self.apiconfigs: Dict[str, Dict[str, Any]] = {
            "binance": {
                "base_url": "https://api.binance.com/api/v3",
                "price_url": "/ticker/price",
                "volume_url": "/ticker/24hr",
                "rate_limit": 1200,
                "weight": 1.0,
                "success_rate": 1.0,
            },
            "coingecko": {
                "base_url": "https://api.coingecko.com/api/v3",
                "price_url": "/simple/price",
                "historical_url": "/coins/{id}/market_chart",
                "volume_url": "/coins/{id}/market_chart",
                "api_key": self.configuration.COINGECKO_API_KEY,
                "rate_limit": 50 if self.configuration.COINGECKO_API_KEY else 10,
                "weight": 0.8 if self.configuration.COINGECKO_API_KEY else 0.5,
                "success_rate": 1.0,
            },
            "uniswap_subgraph": {
                "base_url": f"https://gateway.thegraph.com/api/{os.getenv('GRAPH_API_KEY','')}/subgraphs/id/{os.getenv('UNISWAP_V2_SUBGRAPH_ID','')}",
                "rate_limit": 5,
                "weight": 0.3,
                "success_rate": 1.0,
            },
            "dexscreener": {
                "base_url": "https://api.dexscreener.com/latest/dex",
                "rate_limit": 10,
                "weight": 0.3,
                "success_rate": 1.0,
            },
            "coinpaprika": {
                "base_url": "https://api.coinpaprika.com/v1",
                "price_url": "/tickers/{id}",
                "historical_url": "/coins/{id}/ohlcv/historical",
                "volume_url": "/tickers/{id}",
                "rate_limit": 10,
                "weight": 0.3,
                "success_rate": 1.0,
            },
            "coinmarketcap": {
                "base_url": "https://pro-api.coinmarketcap.com/v1",
                "price_url": "/cryptocurrency/quotes/latest",
                "volume_url": "/cryptocurrency/quotes/latest",
                "historical_url": "/cryptocurrency/quotes/historical",
                "api_key": self.configuration.COINMARKETCAP_API_KEY,
                "rate_limit": 333,
                "weight": 0.7,
                "success_rate": 1.0,
            },
            "cryptocompare": {
                "base_url": "https://min-api.cryptocompare.com/data",
                "price_url": "/price",
                "historical_url": "/v2/histoday",
                "volume_url": "/top/exchanges/full",
                "api_key": self.configuration.CRYPTOCOMPARE_API_KEY,
                "rate_limit": 80,
                "weight": 0.6,
                "success_rate": 1.0,
            },
        }
        self.rate_limiters: Dict[str, asyncio.Semaphore] = {
            name: asyncio.Semaphore(cfg["rate_limit"]) for name, cfg in self.apiconfigs.items()
        }
        self.price_cache = TTLCache(maxsize=2000, ttl=300)
        self.volume_cache = TTLCache(maxsize=1000, ttl=900)
        self.prediction_cache = TTLCache(maxsize=1000, ttl=300)
        self.token_address_to_symbol: Dict[str, str] = {}
        self.token_symbol_to_address: Dict[str, str] = {}
        self.symbol_to_api_id: Dict[str, str] = {}

    async def __aenter__(self) -> "APIConfig":
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        if self.session and not self.session.closed:
            await self.session.close()

    async def initialize(self) -> None:
        self.session = aiohttp.ClientSession()
        token_addresses = await self.configuration._load_json_safe(self.configuration.TOKEN_ADDRESSES, "token addresses")
        token_symbols = await self.configuration._load_json_safe(self.configuration.TOKEN_SYMBOLS, "token symbols")
        for sym, addr in token_addresses.items():
            sym_u = sym.upper()
            addr_l = addr.lower()
            self.token_address_to_symbol[addr_l] = sym_u
            self.token_symbol_to_address[sym_u] = addr_l
            if sym_u in token_symbols:
                self.symbol_to_api_id[sym_u] = token_symbols[sym_u]

    def get_token_symbol(self, address: str) -> Optional[str]:
        return self.token_address_to_symbol.get(address.lower()) if address else None

    def get_token_address(self, symbol: str) -> Optional[str]:
        return self.token_symbol_to_address.get(symbol.upper())

    def _normalize_symbol(self, symbol_or_address: str) -> str:
        if symbol_or_address.startswith("0x") and len(symbol_or_address) == 42:
            sym = self.get_token_symbol(symbol_or_address)
            return self.symbol_to_api_id.get(sym, sym.lower()) if sym else symbol_or_address.lower()
        return self.symbol_to_api_id.get(symbol_or_address.upper(), symbol_or_address.upper().lower())

    def _get_api_symbol(self, symbol_or_address: str, api: str) -> str:
        if symbol_or_address.startswith("0x") and len(symbol_or_address) == 42:
            sym = self.get_token_symbol(symbol_or_address)
            if not sym:
                return symbol_or_address.lower()
            if api == "coingecko":
                return self.symbol_to_api_id.get(sym, sym).lower()
            return sym
        if api == "coingecko":
            return self.symbol_to_api_id.get(symbol_or_address.upper(), symbol_or_address.lower())
        return self.symbol_to_api_id.get(symbol_or_address.upper(), symbol_or_address.upper())

    async def make_request(
        self,
        provider: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        limiter = self.rate_limiters[provider]
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        async with limiter:
            for attempt in range(self.MAX_REQUEST_ATTEMPTS):
                try:
                    timeout = aiohttp.ClientTimeout(total=30)
                    async with self.session.get(url, params=params, headers=headers, timeout=timeout) as resp:
                        if resp.status == 429:
                            await asyncio.sleep(self.REQUEST_BACKOFF_FACTOR ** attempt)
                            continue
                        if provider == "binance" and resp.status == 400:
                            data = await resp.json()
                            code = data.get("code")
                            if code in (-2039, -2038, -1034, -1102, -1128, -1121, -2013, -2026, -2011, -1151, -1008):
                                return None
                        resp.raise_for_status()
                        return await resp.json()
                except (aiohttp.ClientError, aiohttp.ClientResponseError):
                    await asyncio.sleep(self.REQUEST_BACKOFF_FACTOR ** attempt)
                except Exception:
                    await asyncio.sleep(self.REQUEST_BACKOFF_FACTOR ** attempt)
        return None

    async def _fetch_price(self, provider: str, token: str, vs: str) -> Optional[Decimal]:
        cfg = self.apiconfigs[provider]
        try:
            if provider == "binance":
                symbol = token.upper() + vs.upper()
                url = cfg["base_url"] + cfg["price_url"]
                data = await self.make_request(provider, url, {"symbol": symbol})
                return Decimal(data["price"]) if data and "price" in data else None
            if provider == "coingecko":
                token_id = self.symbol_to_api_id.get(token.upper(), token)
                url = cfg["base_url"] + cfg["price_url"]
                params = {"ids": token_id, "vs_currencies": vs}
                headers = {"x-cg-pro-api-key": cfg["api_key"]} if cfg["api_key"] else None
                data = await self.make_request(provider, url, params, headers)
                return Decimal(str(data[token_id][vs])) if data and token_id in data else None
            if provider == "dexscreener":
                url = f"{cfg['base_url']}/pairs/ethereum/{self.get_token_address(token)}"
                data = await self.make_request(provider, url)
                price = data.get("pair", {}).get("priceUsd") if data else None
                return Decimal(str(price)) if price else None
            if provider == "coinpaprika":
                token_id = self.symbol_to_api_id.get(token.upper(), token)
                url = cfg["base_url"] + cfg["price_url"].format(id=token_id)
                data = await self.make_request(provider, url)
                price = data.get("quotes", {}).get(vs.upper(), {}).get("price") if data else None
                return Decimal(str(price)) if price else None
            if provider == "coinmarketcap":
                url = cfg["base_url"] + cfg["price_url"]
                params = {"symbol": token.upper()}
                headers = {"X-CMC_PRO_API_KEY": cfg["api_key"]}
                data = await self.make_request(provider, url, params, headers)
                price = data["data"][token.upper()]["quote"][vs.upper()]["price"] if data else None
                return Decimal(str(price)) if price is not None else None
            if provider == "cryptocompare":
                url = cfg["base_url"] + cfg["price_url"]
                params = {"fsym": token.upper(), "tsyms": vs.upper(), "api_key": cfg["api_key"]}
                data = await self.make_request(provider, url, params)
                return Decimal(str(data[vs.upper()])) if data and vs.upper() in data else None
        except Exception:
            return None
        return None

    async def get_real_time_price(self, token_or_addr: str, vs: str = "usd") -> Optional[Decimal]:
        token_norm = self._normalize_symbol(token_or_addr)
        cache_key = f"price:{token_norm}:{vs}"
        if cache_key in self.price_cache:
            return self.price_cache[cache_key]
        prices = []
        weights = []
        for provider, cfg in self.apiconfigs.items():
            p = await self._fetch_price(provider, token_norm, vs)
            if p is not None:
                prices.append(p)
                weights.append(cfg["weight"] * cfg["success_rate"])
        if not prices:
            return None
        wavg = sum(p * w for p, w in zip(prices, weights)) / sum(weights)
        result = Decimal(str(wavg))
        self.price_cache[cache_key] = result
        return result

    async def _fetch_historical_prices(self, provider: str, token: str, days: int) -> Optional[List[float]]:
        cfg = self.apiconfigs[provider]
        try:
            if provider == "coingecko":
                token_id = self.symbol_to_api_id.get(token.upper(), token)
                url = f"{cfg['base_url']}/coins/{token_id}/market_chart"
                params = {"vs_currency": "usd", "days": days}
                headers = {"x-cg-pro-api-key": cfg["api_key"]} if cfg["api_key"] else None
                data = await self.make_request(provider, url, params, headers)
                return [float(p[1]) for p in data.get("prices", [])] if data else None
            if provider == "cryptocompare":
                url = cfg["base_url"] + cfg["historical_url"]
                params = {"fsym": token.upper(), "tsym": "USD", "limit": days - 1, "api_key": cfg["api_key"]}
                data = await self.make_request(provider, url, params)
                if data and data.get("Response") != "Error":
                    return [float(i["close"]) for i in data["Data"]["Data"]]
            if provider == "dexscreener":
                url = f"{cfg['base_url']}/pairs/ethereum/{self.get_token_address(token)}"
                data = await self.make_request(provider, url)
                chart = data.get("pair", {}).get("priceChart", []) if data else []
                return [float(pt["priceUsd"]) for pt in chart[-days:]] if chart else None
            if provider == "coinpaprika":
                token_id = self.symbol_to_api_id.get(token.upper(), token)
                url = f"{cfg['base_url']}/coins/{token_id}/ohlcv/historical"
                params = {"limit": days}
                data = await self.make_request(provider, url, params)
                return [float(x["close"]) for x in data[-days:]] if isinstance(data, list) else None
        except Exception:
            return None
        return None

    async def fetch_historical_prices(self, token_or_addr: str, days: int = 30) -> List[float]:
        token_norm = self._normalize_symbol(token_or_addr)
        cache_key = f"hist:{token_norm}:{days}"
        if cache_key in self.price_cache:
            return self.price_cache[cache_key]
        for provider in self.apiconfigs:
            hp = await self._fetch_historical_prices(provider, token_norm, days)
            if hp:
                self.price_cache[cache_key] = hp
                return hp
        return []

    async def _fetch_token_volume(self, provider: str, token: str) -> Optional[float]:
        cfg = self.apiconfigs[provider]
        try:
            if provider == "binance":
                symbol = f"{token.upper()}USDT"
                url = cfg["base_url"] + cfg["volume_url"]
                data = await self.make_request(provider, url, {"symbol": symbol})
                return float(data["quoteVolume"]) if data else None
            if provider == "coingecko":
                token_id = self.symbol_to_api_id.get(token.upper(), token)
                url = f"{cfg['base_url']}/coins/{token_id}/market_chart"
                params = {"vs_currency": "usd", "days": 1}
                headers = {"x-cg-pro-api-key": cfg["api_key"]} if cfg["api_key"] else None
                data = await self.make_request(provider, url, params, headers)
                if data and data.get("total_volumes"):
                    return float(data["total_volumes"][-1][1])
            if provider == "cryptocompare":
                url = cfg["base_url"] + cfg["volume_url"]
                params = {"fsym": token.upper(), "tsym": "USD", "api_key": cfg["api_key"]}
                data = await self.make_request(provider, url, params)
                agg = data.get("Data", {}).get("AggregatedData", {}) if data else {}
                return float(agg.get("TOTALVOLUME24H", 0)) if agg else None
            if provider == "dexscreener":
                url = f"{cfg['base_url']}/pairs/ethereum/{self.get_token_address(token)}"
                data = await self.make_request(provider, url)
                vol = data.get("pair", {}).get("volume24hUsd") if data else None
                return float(vol) if vol else None
            if provider == "coinpaprika":
                token_id = self.symbol_to_api_id.get(token.upper(), token)
                url = f"{cfg['base_url']}/tickers/{token_id}"
                data = await self.make_request(provider, url)
                vol = data.get("quotes", {}).get("USD", {}).get("volume_24h") if data else None
                return float(vol) if vol is not None else None
            if provider == "coinmarketcap":
                url = cfg["base_url"] + cfg["volume_url"]
                params = {"symbol": token.upper()}
                headers = {"X-CMC_PRO_API_KEY": cfg["api_key"]}
                data = await self.make_request(provider, url, params, headers)
                vol = data["data"][token.upper()]["quote"]["USD"]["volume_24h"] if data else None
                return float(vol) if vol is not None else None
        except Exception:
            return None
        return None

    async def get_token_volume(self, token_or_addr: str) -> float:
        token_norm = self._normalize_symbol(token_or_addr)
        cache_key = f"vol:{token_norm}"
        if cache_key in self.volume_cache:
            return self.volume_cache[cache_key]
        for provider in self.apiconfigs:
            v = await self._fetch_token_volume(provider, token_norm)
            if v:
                self.volume_cache[cache_key] = v
                return v
        return 0.0

    async def get_token_price_data(
        self, token: str, data_type: str = "current", timeframe: int = 1, vs: str = "usd"
    ) -> Union[Decimal, List[float]]:
        token_norm = self._normalize_symbol(token)
        key = f"{data_type}:{token_norm}:{timeframe}:{vs}"
        if key in self.price_cache:
            return self.price_cache[key]
        if data_type == "current":
            data = await self.get_real_time_price(token_norm, vs)
        elif data_type == "historical":
            data = await self.fetch_historical_prices(token_norm, timeframe)
        else:
            raise ValueError("data_type must be 'current' or 'historical'")
        if data:
            self.price_cache[key] = data
        return data

    async def predict_price(self, token: str) -> float:
        model_file = Path(self.configuration.MODEL_PATH)
        if not model_file.exists():
            hist = await self.fetch_historical_prices(token, 7)
            return float(sum(hist) / len(hist)) if hist else 0.0
        mdl = joblib.load(model_file)
        hist = await self.fetch_historical_prices(token, 7)
        if not hist:
            return 0.0
        df = pd.DataFrame([{
            "price_usd": sum(hist) / len(hist),
            "volume_24h": await self.get_token_volume(token),
            "market_cap": 0,
            "volatility": 0,
            "liquidity_ratio": 0,
            "price_momentum": 0,
        }])
        return float(mdl.predict(df)[0])
