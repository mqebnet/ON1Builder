# python/apiconfig.py

import asyncio
import aiohttp
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from decimal import Decimal
from cachetools import TTLCache
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

from configuration import Configuration
from loggingconfig import setup_logging
import logging

logger = setup_logging("APIConfig", level=logging.INFO)

class APIConfig:
    MAX_REQUEST_ATTEMPTS: int = 5
    REQUEST_BACKOFF_FACTOR: float = 1.5

    def __init__(self, configuration: Configuration):
        self.configuration: Configuration = configuration
        self.session: Optional[aiohttp.ClientSession] = None

        # Define API configurations.
        self.apiconfigs: Dict[str, Dict[str, Any]] = {
            "binance": {
                "base_url": "https://api.binance.com/api/v3",
                "market_url": "/ticker/24hr",
                "rate_limit": 1200,
                "weight": 1.0,
                "success_rate": 1.0,
            },
            "coingecko": {
                "base_url": "https://api.coingecko.com/api/v3",
                "market_url": "/coins/{id}/market_chart",
                "api_key": self.configuration.COINGECKO_API_KEY,
                "rate_limit": 50,
                "weight": 0.8,
                "success_rate": 1.0,
            },
            "coinmarketcap": {
                "base_url": "https://pro-api.coinmarketcap.com/v1",
                "ticker_url": "/cryptocurrency/quotes/latest",
                "api_key": self.configuration.COINMARKETCAP_API_KEY,
                "rate_limit": 333,
                "weight": 0.7,
                "success_rate": 1.0,
            },
            "cryptocompare": {
                "base_url": "https://min-api.cryptocompare.com/data",
                "price_url": "/price",
                "api_key": self.configuration.CRYPTOCOMPARE_API_KEY,
                "rate_limit": 80,
                "weight": 0.6,
                "success_rate": 1.0,
            },
        }
        self.rate_limiters: Dict[str, asyncio.Semaphore] = {
            source: asyncio.Semaphore(config["rate_limit"]) for source, config in self.apiconfigs.items()
        }

        self.price_cache: TTLCache = TTLCache(maxsize=2000, ttl=300)
        self.volume_cache: TTLCache = TTLCache(maxsize=1000, ttl=900)
        self.prediction_cache: TTLCache = TTLCache(maxsize=1000, ttl=300)

        self.token_address_to_symbol: Dict[str, str] = {}
        self.token_symbol_to_address: Dict[str, str] = {}
        self.symbol_to_api_id: Dict[str, str] = {}

    async def __aenter__(self) -> "APIConfig":
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        if self.session:
            await self.session.close()
            logger.debug("APIConfig session closed.")

    async def initialize(self) -> None:
        """Initialize API configuration by loading token addresses and symbols."""
        self.session = aiohttp.ClientSession()
        token_addresses = await self.configuration._load_json_safe(self.configuration.TOKEN_ADDRESSES, "token addresses")
        token_symbols = await self.configuration._load_json_safe(self.configuration.TOKEN_SYMBOLS, "token symbols")
        for symbol, address in token_addresses.items():
            normalized_symbol = symbol.upper()
            normalized_address = address.lower()
            self.token_address_to_symbol[normalized_address] = normalized_symbol
            self.token_symbol_to_address[normalized_symbol] = normalized_address
            if normalized_symbol in token_symbols:
                self.symbol_to_api_id[normalized_symbol] = token_symbols[normalized_symbol]
        logger.debug(f"APIConfig initialized with {len(self.token_address_to_symbol)} tokens.")

    async def close(self) -> None:
        if self.session:
            await self.session.close()

    def get_token_symbol(self, address: str) -> Optional[str]:
        return self.token_address_to_symbol.get(address.lower())

    def get_token_address(self, symbol: str) -> Optional[str]:
        return self.token_symbol_to_address.get(symbol.upper())

    def _normalize_symbol(self, symbol: str) -> str:
        symbol = symbol.upper().strip()
        return self.symbol_to_api_id.get(symbol, symbol.lower())

    def _get_api_symbol(self, symbol: str, api: str) -> str:
        normalized = symbol.upper()
        api_id = self.symbol_to_api_id.get(normalized)
        if api_id and api == "coingecko":
            return api_id
        return normalized

    async def get_token_metadata(self, token: str) -> Optional[Dict[str, Any]]:
        if token.startswith("0x"):
            token = self.token_address_to_symbol.get(token.lower())
            if not token:
                logger.warning(f"No symbol found for token address: {token}")
                return None
        normalized_symbol = self._normalize_symbol(token)
        cache_key = f"metadata_{normalized_symbol}"
        try:
            token_id = normalized_symbol.lower()
            url = f"{self.apiconfigs['coingecko']['base_url']}/coins/{token_id}"
            headers = {"x-cg-pro-api-key": self.apiconfigs["coingecko"]["api_key"]} if self.apiconfigs["coingecko"]["api_key"] else None
            response = await self.make_request("coingecko", url, headers=headers)
            if response:
                metadata = {
                    "symbol": response.get("symbol", ""),
                    "market_cap": response.get("market_data", {}).get("market_cap", {}).get("usd", 0),
                    "total_supply": response.get("market_data", {}).get("total_supply", 0),
                    "circulating_supply": response.get("market_data", {}).get("circulating_supply", 0),
                    "trading_pairs": len(response.get("tickers", [])),
                    "exchanges": list({t.get("market", {}).get("name") for t in response.get("tickers", []) if t.get("market", {}).get("name")})
                }
                return metadata
            return None
        except Exception as e:
            logger.error(f"Error fetching metadata for {normalized_symbol}: {e}", exc_info=True)
            return None

    async def get_real_time_price(self, token: str, vs_currency: str = "eth") -> Optional[Decimal]:
        if token.startswith("0x"):
            token = self.token_address_to_symbol.get(token.lower())
            if not token:
                logger.warning("Token symbol unavailable.")
                return None
        normalized_symbol = self._normalize_symbol(token)
        cache_key = f"price_{normalized_symbol}_{vs_currency}"
        if cache_key in self.price_cache:
            return self.price_cache[cache_key]
        prices = []
        weights = []
        for source, config in self.apiconfigs.items():
            try:
                price = await self._fetch_price(source, normalized_symbol, vs_currency)
                if price is not None:
                    prices.append(price)
                    weights.append(config["weight"] * config["success_rate"])
            except Exception as e:
                logger.error(f"Error fetching price from {source}: {e}")
                config["success_rate"] *= 0.9
        if not prices:
            logger.warning(f"No valid price for {token}.")
            return None
        weighted_price = sum(p * w for p, w in zip(prices, weights)) / sum(weights)
        self.price_cache[cache_key] = Decimal(str(weighted_price))
        return self.price_cache[cache_key]

    async def _fetch_price(self, source: str, token: str, vs_currency: str) -> Optional[Decimal]:
        config = self.apiconfigs.get(source)
        if not config:
            logger.error(f"API source {source} not configured.")
            return None
        try:
            if source == "binance":
                symbol = self._get_api_symbol(token, source) + "USDT"
                url = f"{config['base_url']}{config['market_url']}"
                params = {"symbol": symbol}
            elif source == "coingecko":
                token_id = token.lower()
                url = f"{config['base_url']}/simple/price"
                params = {"ids": token_id, "vs_currencies": vs_currency}
            elif source == "coinmarketcap":
                url = f"{config['base_url']}{config['ticker_url']}"
                headers = {"X-CMC_PRO_API_KEY": config["api_key"], "Accept": "application/json"}
                params = {"symbol": token.upper()}
            elif source == "cryptocompare":
                url = f"{config['base_url']}{config['price_url']}"
                params = {"fsym": token.upper(), "tsyms": vs_currency.upper()}
            else:
                logger.error(f"Unsupported source: {source}")
                return None
            response = await self.make_request(source, url, params=params)
            if source == "binance":
                if "lastPrice" in response:
                    return Decimal(response["lastPrice"])
            elif source == "coingecko":
                if token.lower() in response and vs_currency in response[token.lower()]:
                    return Decimal(str(response[token.lower()][vs_currency]))
            elif source == "coinmarketcap":
                if "data" in response and token.upper() in response["data"]:
                    data = response["data"][token.upper()]
                    price = data.get("quote", {}).get(vs_currency.upper(), {}).get("price")
                    if price is not None:
                        return Decimal(str(price))
            elif source == "cryptocompare":
                if vs_currency.upper() in response:
                    return Decimal(str(response[vs_currency.upper()]))
            return None
        except Exception as e:
            logger.error(f"Error fetching price from {source}: {e}", exc_info=True)
            return None

    async def make_request(self,
                           provider_name: str,
                           url: str,
                           params: Optional[Dict[str, Any]] = None,
                           headers: Optional[Dict[str, str]] = None) -> Any:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        rate_limiter = self.rate_limiters.get(provider_name)
        if not rate_limiter:
            logger.error(f"No rate limiter for {provider_name}")
            return None
        async with rate_limiter:
            for attempt in range(self.MAX_REQUEST_ATTEMPTS):
                try:
                    timeout = aiohttp.ClientTimeout(total=30)
                    async with self.session.get(url, params=params, headers=headers, timeout=timeout) as response:
                        if response.status == 429:
                            await asyncio.sleep(self.REQUEST_BACKOFF_FACTOR ** attempt)
                            continue
                        response.raise_for_status()
                        return await response.json()
                except Exception as e:
                    logger.warning(f"Attempt {attempt+1} for {provider_name} failed: {e}")
                    await asyncio.sleep(self.REQUEST_BACKOFF_FACTOR ** attempt)
            logger.error(f"All attempts failed for {provider_name}")
            return None

    async def fetch_historical_prices(self, token: str, days: int = 30) -> List[float]:
        cache_key = f"historical_{token}_{days}"
        if cache_key in self.price_cache:
            return self.price_cache[cache_key]
        prices = await self._fetch_from_services(
            lambda service: self._fetch_historical_prices(service, token.lower(), days),
            f"historical prices for {token}"
        )
        if prices:
            self.price_cache[cache_key] = prices
        return prices or []

    async def _fetch_historical_prices(self, source: str, token: str, days: int) -> Optional[List[float]]:
        config = self.apiconfigs.get(source)
        if not config:
            logger.error(f"API source {source} not configured for historical prices.")
            return None
        try:
            url = f"{config['base_url']}/coins/{token}/market_chart"
            params = {"vs_currency": "eth", "days": days}
            async with self.session.get(url, params=params, timeout=30) as response:
                response.raise_for_status()
                data = await response.json()
                prices = [price_point[1] for price_point in data.get("prices", [])]
                logger.debug(f"Fetched historical prices from {source}.")
                return prices
        except Exception as e:
            logger.error(f"Error fetching historical prices from {source}: {e}", exc_info=True)
            return None

    async def _fetch_from_services(self, fetch_func: callable, description: str) -> Optional[Union[List[float], float]]:
        for source in self.apiconfigs.keys():
            try:
                logger.debug(f"Fetching {description} using {source}...")
                result = await fetch_func(source)
                if result:
                    return result
            except Exception as e:
                logger.warning(f"Failed to fetch {description} from {source}: {e}")
        logger.warning(f"Failed to fetch {description}.")
        return None

    async def get_token_volume(self, token: str) -> float:
        cache_key = f"volume_{token}"
        if cache_key in self.volume_cache:
            return self.volume_cache[cache_key]
        volume = await self._fetch_from_services(
            lambda source: self._fetch_token_volume(source, token),
            f"trading volume for {token}"
        )
        if volume is not None:
            self.volume_cache[cache_key] = volume
        return volume or 0.0

    async def _fetch_token_volume(self, source: str, token: str) -> Optional[float]:
        config = self.apiconfigs.get(source)
        if not config:
            return None
        try:
            if source == "binance":
                symbol = f"{token}USDT"
                url = f"{config['base_url']}{config['market_url']}"
                params = {"symbol": symbol}
                response = await self.make_request(source, url, params=params)
                if response and "quoteVolume" in response:
                    return float(response["quoteVolume"])
            elif source == "coingecko":
                token_id = token.lower()
                url = f"{config['base_url']}/simple/price"
                params = {"ids": token_id, "vs_currencies": "usd", "include_24hr_vol": "true"}
                response = await self.make_request(source, url, params=params)
                if response and token_id in response:
                    return float(response[token_id].get("usd_24h_vol", 0))
            elif source == "coinmarketcap":
                url = f"{config['base_url']}{config['ticker_url']}"
                headers = {"X-CMC_PRO_API_KEY": config["api_key"], "Accept": "application/json"}
                params = {"symbol": token.upper()}
                response = await self.make_request(source, url, params=params, headers=headers)
                if response and "data" in response:
                    data = response["data"].get(token.upper(), {})
                    return float(data.get("quote", {}).get("USD", {}).get("volume_24h", 0))
            return None
        except Exception as e:
            logger.error(f"Error fetching token volume from {source}: {e}", exc_info=True)
            return None

    async def get_token_price_data(self, token_symbol: str, data_type: str = "current",
                                   timeframe: int = 1, vs_currency: str = "eth") -> Union[float, List[float]]:
        cache_key = f"{data_type}_{token_symbol}_{timeframe}_{vs_currency}"
        if cache_key in self.price_cache:
            return self.price_cache[cache_key]
        try:
            if data_type == "current":
                data = await self.get_real_time_price(token_symbol, vs_currency)
            elif data_type == "historical":
                data = await self.fetch_historical_prices(token_symbol, days=timeframe)
            else:
                raise ValueError(f"Invalid data type: {data_type}")
            if data is not None:
                self.price_cache[cache_key] = data
            return data
        except Exception as e:
            logger.error(f"Error fetching token price data for {token_symbol}: {e}", exc_info=True)
            return [] if data_type == "historical" else 0.0

    async def get_token_volume_data(self, token_symbol: str) -> float:
        return await self.get_token_volume(token_symbol)

    async def update_training_data(self) -> None:
        logger.info("Updating training data... (Not implemented)")
        # Implement updating training data if required.

    async def train_price_model(self) -> None:
        try:
            training_data_path = self.configuration.TRAINING_DATA_PATH
            model_path = self.configuration.MODEL_PATH
            if not Path(training_data_path).exists():
                logger.warning("No training data found for model training.")
                return
            df = pd.read_csv(training_data_path)
            if len(df) < self.configuration.MIN_TRAINING_SAMPLES:
                logger.warning(f"Insufficient training samples: {len(df)}")
                return
            features = ["price_usd", "volume_24h", "market_cap", "volatility", "liquidity_ratio", "price_momentum"]
            X = df[features].fillna(0)
            y = df["price_usd"].fillna(0)
            model = LinearRegression()
            model.fit(X, y)
            joblib.dump(model, model_path)
            logger.info(f"Trained price model and saved to {model_path}")
        except Exception as e:
            logger.error(f"Error training price model: {e}")

    async def predict_price(self, token: str) -> float:
        try:
            model_path = Path(self.configuration.MODEL_PATH)
            if not model_path.exists():
                await self.train_price_model()
                if not model_path.exists():
                    return 0.0
            model = joblib.load(model_path)
            market_data = await self.get_token_price_data(token, "historical", timeframe=7)
            if not market_data or not isinstance(market_data, list):
                return 0.0
            df_features = pd.DataFrame([{
                "price_usd": sum(market_data) / len(market_data),
                "volume_24h": await self.get_token_volume_data(token),
                "market_cap": 0,
                "volatility": 0,
                "liquidity_ratio": 0,
                "price_momentum": 0
            }])
            prediction = model.predict(df_features)[0]
            return float(prediction)
        except Exception as e:
            logger.error(f"Error predicting price for {token}: {e}")
            return 0.0
