# -*- coding: utf-8 -*-
"""
APIConfig Module

Provides cryptocurrency data integration from multiple sources (Binance, CoinGecko,
CoinMarketCap, CryptoCompare). This module handles rate limiting, caching, endpoint
construction, and multi-source aggregation for price, volume, and market data.
"""

import asyncio
import json
import time
import aiohttp
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union
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
    """
    Cryptocurrency API Integration and Data Management System.

    Attributes:
        MAX_REQUEST_ATTEMPTS: Maximum number of retry attempts for HTTP requests.
        REQUEST_BACKOFF_FACTOR: Multiplier for exponential backoff between retries.
    """

    MAX_REQUEST_ATTEMPTS: int = 5
    REQUEST_BACKOFF_FACTOR: float = 1.5

    def __init__(self, configuration: Configuration):
        """
        Initialize the APIConfig with a configuration instance.

        Args:
            configuration (Configuration): The configuration object containing API keys and paths.
        """
        self.configuration: Configuration = configuration
        self.session: Optional[aiohttp.ClientSession] = None
        self.price_cache: TTLCache = TTLCache(maxsize=2000, ttl=300)
        self.volume_cache: TTLCache = TTLCache(maxsize=1000, ttl=900)
        self.market_data_cache: TTLCache = TTLCache(maxsize=1000, ttl=1800)
        self.token_metadata_cache: TTLCache = TTLCache(maxsize=500, ttl=86400)

        self.rate_limit_counters: Dict[str, Dict[str, Any]] = {
            "coingecko": {"count": 0, "reset_time": time.time(), "limit": 50},
            "coinmarketcap": {"count": 0, "reset_time": time.time(), "limit": 330},
            "cryptocompare": {"count": 0, "reset_time": time.time(), "limit": 80},
            "binance": {"count": 0, "reset_time": time.time(), "limit": 1200},
        }

        self.high_priority_tokens: set[str] = set()
        self.update_intervals: Dict[str, int] = {
            'price': 30,
            'volume': 300,
            'market_data': 1800,
            'metadata': 86400
        }

        self.session = None

        # API configuration for various providers
        self.apiconfigs: Dict[str, Dict[str, Any]] = {
            "binance": {
                "base_url": "https://api.binance.com/api/v3",
                "market_url": "/ticker/24hr",
                "success_rate": 1.0,
                "weight": 1.0,
                "rate_limit": 1200,
            },
            "coingecko": {
                "base_url": "https://api.coingecko.com/api/v3",
                # For historical data, endpoint is different from current price
                "market_url": "/coins/{id}/market_chart",
                "volume_url": "/coins/{id}",
                "api_key": configuration.COINGECKO_API_KEY,
                "success_rate": 1.0,
                "weight": 0.8,
                "rate_limit": 50,
            },
            "coinmarketcap": {
                "base_url": "https://pro-api.coinmarketcap.com/v1",
                "ticker_url": "/cryptocurrency/quotes/latest",
                "api_key": configuration.COINMARKETCAP_API_KEY,
                "success_rate": 1.0,
                "weight": 0.7,
                "rate_limit": 333,
            },
            "cryptocompare": {
                "base_url": "https://min-api.cryptocompare.com/data",
                "price_url": "/price",
                "api_key": configuration.CRYPTOCOMPARE_API_KEY,
                "success_rate": 1.0,
                "weight": 0.6,
                "rate_limit": 80,
            },
        }

        self.rate_limiters: Dict[str, asyncio.Semaphore] = {
            provider: asyncio.Semaphore(config.get("rate_limit", 10))
            for provider, config in self.apiconfigs.items()
        }

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
        """
        Initialize API configuration by loading token addresses and symbols from configuration.
        """
        try:
            self.session = aiohttp.ClientSession()

            token_addresses = await self.configuration._load_json_safe(
                self.configuration.TOKEN_ADDRESSES,
                "token addresses"
            )
            token_symbols = await self.configuration._load_json_safe(
                self.configuration.TOKEN_SYMBOLS,
                "token symbols"
            )

            self.token_address_to_symbol = {}
            self.token_symbol_to_address = {}
            self.symbol_to_api_id = {}

            for symbol, address in token_addresses.items():
                normalized_symbol = symbol.upper()
                normalized_address = address.lower()

                self.token_address_to_symbol[normalized_address] = normalized_symbol
                self.token_symbol_to_address[normalized_symbol] = normalized_address

                if normalized_symbol in token_symbols:
                    self.symbol_to_api_id[normalized_symbol] = token_symbols[normalized_symbol]

            logger.debug(f"APIConfig initialized with {len(self.token_address_to_symbol)} tokens ✅")

        except Exception as e:
            logger.critical(f"APIConfig initialization failed: {e}")
            raise

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()

    def get_token_symbol(self, address: str) -> Optional[str]:
        """
        Get the token symbol for a given address.

        Args:
            address (str): The token contract address.

        Returns:
            Optional[str]: The token symbol, or None if not found.
        """
        return self.token_address_to_symbol.get(address.lower())

    def get_token_address(self, symbol: str) -> Optional[str]:
        """
        Get the token address for a given symbol.

        Args:
            symbol (str): The token symbol.

        Returns:
            Optional[str]: The token address, or None if not found.
        """
        return self.token_symbol_to_address.get(symbol.upper())

    def _normalize_symbol(self, symbol: str) -> str:
        """
        Normalize a token symbol and return its API-specific identifier if available.

        Args:
            symbol (str): The token symbol.

        Returns:
            str: The normalized token symbol.
        """
        symbol = symbol.upper().strip()
        return self.symbol_to_api_id.get(symbol, symbol.lower())

    def _get_api_symbol(self, symbol: str, api: str) -> str:
        """
        Get the API-specific symbol format.

        Args:
            symbol (str): The token symbol.
            api (str): The API provider name.

        Returns:
            str: The formatted symbol.
        """
        normalized = symbol.upper()
        api_id = self.symbol_to_api_id.get(normalized)
        if api_id:
            if api == "coingecko":
                return api_id
            # For Binance, CoinMarketCap, and CryptoCompare, use the normalized symbol.
            return normalized
        return normalized

    async def get_token_metadata(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve token metadata using the token symbol (or address).

        Args:
            token (str): The token symbol or address.

        Returns:
            Optional[Dict[str, Any]]: The metadata dictionary, or None if not found.
        """
        if token.startswith('0x'):
            token = self.token_address_to_symbol.get(token.lower())
            if not token:
                logger.warning(f"No symbol found for address {token}")
                return None

        normalized_symbol = self._normalize_symbol(token)

        if normalized_symbol in self.token_metadata_cache:
            return self.token_metadata_cache[normalized_symbol]

        metadata = await self._fetch_from_services(
            lambda service: self._fetch_token_metadata(service, normalized_symbol),
            f"metadata for {normalized_symbol}",
        )

        if metadata:
            self.token_metadata_cache[normalized_symbol] = metadata
        return metadata

    async def _fetch_token_metadata(self, source: str, token: str) -> Optional[Dict[str, Any]]:
        """
        Fetch token metadata from a specific API source.

        Args:
            source (str): The API provider.
            token (str): The token symbol.

        Returns:
            Optional[Dict[str, Any]]: The metadata, or None if fetching fails.
        """
        config = self.apiconfigs.get(source)
        if not config:
            return None

        try:
            if source == "coingecko":
                token_id = token.lower()
                url = f"{config['base_url']}/coins/{token_id}"
                headers = {"x-cg-pro-api-key": config['api_key']} if config['api_key'] else None
                response = await self.make_request(source, url, headers=headers)
                if response:
                    return {
                        'symbol': response.get('symbol', ''),
                        'market_cap': response.get('market_data', {}).get('market_cap', {}).get('usd', 0),
                        'total_supply': response.get('market_data', {}).get('total_supply', 0),
                        'circulating_supply': response.get('market_data', {}).get('circulating_supply', 0),
                        'trading_pairs': len(response.get('tickers', [])),
                        'exchanges': list(set(t.get('market', {}).get('name') for t in response.get('tickers', [])))
                    }

            elif source == "coinmarketcap":
                url = f"{config['base_url']}{config['ticker_url']}"
                headers = {
                    "X-CMC_PRO_API_KEY": config['api_key'],
                    "Accept": "application/json"
                }
                params = {"symbol": token.upper()}
                response = await self.make_request(source, url, params=params, headers=headers)
                if response and 'data' in response:
                    data = response['data'].get(token.upper(), {})
                    return {
                        'symbol': data.get('symbol', ''),
                        'market_cap': data.get('quote', {}).get('USD', {}).get('market_cap', 0),
                        'total_supply': data.get('total_supply', 0),
                        'circulating_supply': data.get('circulating_supply', 0),
                        'trading_pairs': len(data.get('tags', [])),
                        'exchanges': []
                    }
            return None

        except aiohttp.ClientError as e:
            logger.error(f"aiohttp ClientError fetching metadata from {source}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching metadata from {source}: {e}", exc_info=True)
            return None

    async def get_real_time_price(self, token: str, vs_currency: str = "eth") -> Optional[Decimal]:
        """
        Fetch the real-time price of a token using multi-source aggregation.

        Args:
            token (str): The token symbol or address.
            vs_currency (str): The currency to compare against (default "eth").

        Returns:
            Optional[Decimal]: The token price in the specified currency, or None if unavailable.
        """
        if token.startswith('0x'):
            token = self.token_address_to_symbol.get(token.lower())
            if not token:
                logger.warning(f"No symbol found for address {token}")
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
                if price:
                    prices.append(price)
                    weights.append(config["weight"] * config["success_rate"])
            except Exception as e:
                logger.error(f"Error fetching price from {source}: {e}")
                config["success_rate"] *= 0.9
        if not prices:
            logger.warning(f"No valid prices found for {token}!")
            return None
        weighted_price = sum(p * w for p, w in zip(prices, weights)) / sum(weights)
        self.price_cache[cache_key] = Decimal(str(weighted_price))
        return self.price_cache[cache_key]

    async def _fetch_price(self, source: str, token: str, vs_currency: str) -> Optional[Decimal]:
        """
        Fetch the price of a token from a specified API source.

        Args:
            source (str): The API provider.
            token (str): The token symbol.
            vs_currency (str): The target currency.

        Returns:
            Optional[Decimal]: The fetched token price, or None on failure.
        """
        config = self.apiconfigs.get(source)
        if not config:
            logger.error(f"API source {source} not configured.")
            return None

        try:
            if source == "binance":
                # Binance uses the /ticker/24hr endpoint. Assume USDT pair.
                symbol = self._get_api_symbol(token, source) + "USDT"
                url = f"{config['base_url']}{config['market_url']}"
                params = {"symbol": symbol}
            elif source == "coingecko":
                token_id = token.lower()
                url = f"{config['base_url']}/simple/price"
                params = {"ids": token_id, "vs_currencies": vs_currency}
            elif source == "coinmarketcap":
                url = f"{config['base_url']}{config['ticker_url']}"
                headers = {
                    "X-CMC_PRO_API_KEY": config['api_key'],
                    "Accept": "application/json"
                }
                params = {"symbol": token.upper()}
            elif source == "cryptocompare":
                url = f"{config['base_url']}{config['price_url']}"
                params = {"fsym": token.upper(), "tsyms": vs_currency.upper()}
            else:
                logger.error(f"Unsupported source: {source}")
                return None

            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                if source == "binance":
                    if "lastPrice" in data:
                        return Decimal(data["lastPrice"])
                elif source == "coingecko":
                    if token.lower() in data and vs_currency in data[token.lower()]:
                        return Decimal(str(data[token.lower()][vs_currency]))
                elif source == "coinmarketcap":
                    if "data" in data and token.upper() in data["data"]:
                        price = data["data"][token.upper()].get("quote", {}).get(vs_currency.upper(), {}).get("price")
                        if price:
                            return Decimal(str(price))
                elif source == "cryptocompare":
                    if vs_currency.upper() in data:
                        return Decimal(str(data[vs_currency.upper()]))
                return None

        except aiohttp.ClientError as e:
            logger.error(f"aiohttp ClientError fetching price from {source}: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSONDecodeError decoding price from {source}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching price from {source}: {e}", exc_info=True)
            return None

    async def make_request(
        self,
        provider_name: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        """
        Make an HTTP GET request with retries and error handling.

        Args:
            provider_name (str): The API provider name.
            url (str): The target URL.
            params (Optional[Dict[str, Any]]): Query parameters.
            headers (Optional[Dict[str, str]]): HTTP headers.

        Returns:
            Any: The JSON-decoded response, or None if the request fails.
        """
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

        rate_limiter = self.rate_limiters.get(provider_name)
        if rate_limiter is None:
            logger.error(f"No rate limiter for provider {provider_name}")
            return None

        async with rate_limiter:
            for attempt in range(self.MAX_REQUEST_ATTEMPTS):
                try:
                    timeout = aiohttp.ClientTimeout(total=30, connect=10, sock_read=10)
                    async with self.session.get(url, params=params, headers=headers, timeout=timeout) as response:
                        if response.status == 429:
                            wait_time = self.REQUEST_BACKOFF_FACTOR ** attempt
                            logger.warning(f"Rate limit for {provider_name}, waiting {wait_time}s")
                            await asyncio.sleep(wait_time)
                            continue

                        response.raise_for_status()
                        return await response.json()

                except aiohttp.ClientResponseError as e:
                    logger.warning(f"ClientResponseError {e} from {provider_name}, attempt {attempt + 1}/{self.MAX_REQUEST_ATTEMPTS}")
                    if attempt == self.MAX_REQUEST_ATTEMPTS - 1:
                        logger.error(f"Max retries reached for {provider_name} after ClientResponseError. Returning None.")
                        return None
                except aiohttp.ClientConnectionError as e:
                    logger.warning(f"ClientConnectionError {e} for {provider_name}, attempt {attempt + 1}/{self.MAX_REQUEST_ATTEMPTS}")
                    if attempt == self.MAX_REQUEST_ATTEMPTS - 1:
                        logger.error(f"Max retries reached for {provider_name} after ClientConnectionError. Returning None.")
                        return None
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout for {provider_name} (attempt {attempt + 1})")
                    if attempt == self.MAX_REQUEST_ATTEMPTS - 1:
                        logger.error(f"Max retries reached for {provider_name} after Timeout. Returning None.")
                        return None
                except json.JSONDecodeError as e:
                    logger.error(f"JSONDecodeError from {provider_name}: {e}")
                    return None
                except Exception as e:
                    logger.error(f"Unexpected error fetching from {provider_name}: {e}", exc_info=True)
                    if attempt == self.MAX_REQUEST_ATTEMPTS - 1:
                        logger.error(f"Max retries reached for {provider_name} after unexpected error. Returning None.")
                        return None

                await asyncio.sleep(self.REQUEST_BACKOFF_FACTOR ** attempt)

            logger.error(f"All attempts failed for {provider_name}. Returning None.")
            return None

    async def fetch_historical_prices(self, token: str, days: int = 30) -> List[float]:
        """
        Fetch historical price data for a token over a specified number of days.

        Args:
            token (str): The token symbol.
            days (int): The number of days of historical data (default: 30).

        Returns:
            List[float]: A list of historical prices.
        """
        cache_key = f"historical_prices_{token}_{days}"
        if cache_key in self.price_cache:
            logger.debug(f"Returning cached historical prices for {token}.")
            return self.price_cache[cache_key]
        prices = await self._fetch_from_services(
            lambda service: self._fetch_historical_prices(service, token.lower(), days),
            f"historical prices for {token}",
        )
        if prices:
            self.price_cache[cache_key] = prices
        return prices or []

    async def _fetch_historical_prices(self, source: str, token: str, days: int) -> Optional[List[float]]:
        """
        Fetch historical prices for a token from a specified API source.

        Args:
            source (str): The API provider.
            token (str): The token symbol (assumed lowercase for CoinGecko).
            days (int): Number of days to fetch.

        Returns:
            Optional[List[float]]: A list of prices, or None if fetching fails.
        """
        config = self.apiconfigs.get(source)
        if not config:
            logger.error(f"API source {source} not configured.")
            return None

        try:
            # For CoinGecko, use the /coins/{id}/market_chart endpoint
            url = f"{config['base_url']}/coins/{token}/market_chart"
            params = {"vs_currency": "eth", "days": days}
            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                prices = [price[1] for price in data.get("prices", [])]
                logger.debug(f"Fetched historical prices from {source}: {prices}")
                return prices
        except aiohttp.ClientError as e:
            logger.error(f"aiohttp ClientError fetching historical prices from {source}: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSONDecodeError decoding historical prices from {source}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching historical prices from {source}: {e}", exc_info=True)
            return None

    async def get_token_volume(self, token: str) -> float:
        """
        Get the 24-hour trading volume for a given token.

        Args:
            token (str): The token symbol.

        Returns:
            float: The trading volume, or 0.0 if unavailable.
        """
        cache_key = f"token_volume_{token}"
        if cache_key in self.volume_cache:
            logger.debug(f"Returning cached trading volume for {token}.")
            return self.volume_cache[cache_key]
        volume = await self._fetch_from_services(
            lambda service: self._fetch_token_volume(service, token),
            f"trading volume for {token}",
        )
        if volume is not None:
            self.volume_cache[cache_key] = volume
        return volume or 0.0

    async def _fetch_token_volume(self, source: str, token: str) -> Optional[float]:
        """
        Fetch the trading volume for a token from a specified source.

        Args:
            source (str): The API provider.
            token (str): The token symbol.

        Returns:
            Optional[float]: The trading volume, or None if fetching fails.
        """
        config = self.apiconfigs.get(source)
        if not config:
            return None

        try:
            if source == "binance":
                symbols = await self._get_trading_pairs(token)
                if not symbols:
                    return None
                for symbol in symbols:
                    try:
                        url = f"{config['base_url']}{config['market_url']}"
                        params = {"symbol": symbol}
                        response = await self.make_request(source, url, params=params)
                        # Use 'quoteVolume' as returned by Binance
                        if response and 'quoteVolume' in response:
                            return float(response['quoteVolume'])
                    except Exception:
                        continue

            elif source == "coingecko":
                token_id = token.lower()
                url = f"{config['base_url']}/simple/price"
                params = {"ids": token_id, "vs_currencies": "usd", "include_24hr_vol": "true"}
                if config['api_key']:
                    params['x_cg_pro_api_key'] = config['api_key']
                response = await self.make_request(source, url, params=params)
                if response and token_id in response:
                    return float(response[token_id].get('usd_24h_vol', 0))

            elif source == "coinmarketcap":
                url = f"{config['base_url']}/cryptocurrency/quotes/latest"
                headers = {"X-CMC_PRO_API_KEY": config['api_key']}
                params = {"symbol": token.upper()}
                response = await self.make_request(source, url, params=params, headers=headers)
                if response and 'data' in response:
                    token_data = response['data'].get(token.upper(), {})
                    return float(token_data.get('quote', {}).get('USD', {}).get('volume_24h', 0))

            return None

        except aiohttp.ClientError as e:
            logger.error(f"aiohttp ClientError fetching volume from {source}: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSONDecodeError decoding volume from {source}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching volume from {source}: {e}", exc_info=True)
            return None

    async def _get_trading_pairs(self, token: str) -> List[str]:
        """
        Get valid trading pairs for a token.

        Args:
            token (str): The token symbol.

        Returns:
            List[str]: A list of trading pair symbols.
        """
        quote_currencies = ["USDT", "BUSD", "USD", "ETH", "BTC"]
        symbol_mappings = {
            "WETH": ["ETH"],
            "WBTC": ["BTC"],
            "ETH": ["ETH"],
            "BTC": ["BTC"],
        }

        base_symbols = symbol_mappings.get(token, [token])
        pairs = []
        for base in base_symbols:
            pairs.extend([f"{base}{quote}" for quote in quote_currencies])
        return pairs

    async def _fetch_from_services(self, fetch_func: Callable[[str], Any], description: str) -> Optional[Union[List[float], float]]:
        """
        Helper method to fetch data from multiple services.

        Args:
            fetch_func (Callable[[str], Any]): A function that fetches data for a given service.
            description (str): A description of the data being fetched.

        Returns:
            Optional[Union[List[float], float]]: The fetched data, or None if all services fail.
        """
        for service in self.apiconfigs.keys():
            try:
                logger.debug(f"Fetching {description} using {service}...")
                result = await fetch_func(service)
                if result:
                    return result
            except Exception as e:
                logger.warning(f"Failed to fetch {description} using {service}: {e}")
        logger.warning(f"Failed to fetch {description}.")
        return None

    async def get_token_price_data(
        self,
        token_symbol: str,
        data_type: str = 'current',
        timeframe: int = 1,
        vs_currency: str = 'eth'
    ) -> Union[float, List[float]]:
        """
        Fetch token price data (current or historical).

        Args:
            token_symbol (str): The token symbol.
            data_type (str): 'current' for latest price or 'historical' for historical data.
            timeframe (int): Number of days for historical data.
            vs_currency (str): The target currency (default: 'eth').

        Returns:
            Union[float, List[float]]: The current price as a float or a list of historical prices.
        """
        cache_key = f"{data_type}_{token_symbol}_{timeframe}_{vs_currency}"
        if cache_key in self.price_cache:
            return self.price_cache[cache_key]
        try:
            if data_type == 'current':
                data = await self.get_real_time_price(token_symbol, vs_currency)
            elif data_type == 'historical':
                data = await self.fetch_historical_prices(token_symbol, days=timeframe)
            else:
                raise ValueError(f"Invalid data type: {data_type}")
            if data is not None:
                self.price_cache[cache_key] = data
            return data
        except Exception as e:
            logger.error(f"Error fetching {data_type} price data: {e}")
            return [] if data_type == 'historical' else 0.0

    def _calculate_volatility(self, price_history: List[float]) -> float:
        """
        Calculate price volatility as the standard deviation of returns.

        Args:
            price_history (List[float]): Historical price data.

        Returns:
            float: The volatility as a decimal fraction.
        """
        if not price_history or len(price_history) < 2:
            return 0.0
        try:
            returns = [
                (price_history[i] - price_history[i - 1]) / price_history[i - 1]
                for i in range(1, len(price_history))
            ]
            return float(np.std(returns))
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.0

    def _calculate_momentum(self, price_history: List[float]) -> float:
        """
        Calculate price momentum using exponential moving averages.

        Args:
            price_history (List[float]): Historical price data.

        Returns:
            float: The calculated momentum.
        """
        if not price_history or len(price_history) < 2:
            return 0.0
        try:
            short_period = min(12, len(price_history))
            long_period = min(26, len(price_history))
            ema_short = sum(price_history[-short_period:]) / short_period
            ema_long = sum(price_history[-long_period:]) / long_period
            momentum = (ema_short / ema_long - 1) if ema_long > 0 else 0
            return float(momentum)
        except Exception as e:
            logger.error(f"Error calculating momentum: {e}")
            return 0.0

    async def _gather_training_data(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Gather training data for a token for model training.

        Args:
            token (str): The token symbol.

        Returns:
            Optional[Dict[str, Any]]: A dictionary with training data, or None if data gathering fails.
        """
        try:
            data = await asyncio.gather(
                self.get_real_time_price(token),
                self.get_token_volume(token),
                self._fetch_market_data(token),
                return_exceptions=True
            )
            if any(isinstance(r, Exception) for r in data):
                logger.warning(f"Error gathering data for {token}")
                return None
            return {
                'timestamp': int(time.time()),
                'symbol': token,
                'price_usd': float(data[0]),
                'volume_24h': float(data[1]),
                **(data[2] if isinstance(data[2], dict) else {})
            }
        except Exception as e:
            logger.error(f"Error gathering training data: {e}")
            return None

    async def _fetch_market_data(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Fetch comprehensive market data for a token.

        Args:
            token (str): The token symbol.

        Returns:
            Optional[Dict[str, Any]]: A dictionary with market data, or None if fetching fails.
        """
        try:
            cache_key = f"market_data_{token}"
            if cache_key in self.market_data_cache:
                return self.market_data_cache[cache_key]
            data_tasks = [
                self.get_token_metadata(token),
                self.get_token_volume(token),
                self.get_token_price_data(token, 'historical', timeframe=7)
            ]
            metadata, volume, price_history = await asyncio.gather(*data_tasks, return_exceptions=True)
            if any(isinstance(r, Exception) for r in (metadata, volume, price_history)):
                logger.warning(f"Some market data fetching failed for {token}")
                return None
            price_volatility = self._calculate_volatility(price_history) if price_history else 0
            market_data = {
                'market_cap': metadata.get('market_cap', 0) if metadata else 0,
                'volume_24h': float(volume) if volume else 0,
                'percent_change_24h': metadata.get('percent_change_24h', 0) if metadata else 0,
                'total_supply': metadata.get('total_supply', 0) if metadata else 0,
                'circulating_supply': metadata.get('circulating_supply', 0) if metadata else 0,
                'volatility': price_volatility,
                'price_momentum': self._calculate_momentum(price_history) if price_history else 0,
                'liquidity_ratio': await self._calculate_liquidity_ratio(token),
                'trading_pairs': len(metadata.get('trading_pairs', [])) if metadata else 0,
                'exchange_count': len(metadata.get('exchanges', [])) if metadata else 0
            }
            self.market_data_cache[cache_key] = market_data
            return market_data
        except Exception as e:
            logger.error(f"Error fetching market data for {token}: {e}")
            return None

    async def _calculate_liquidity_ratio(self, token: str) -> float:
        """
        Calculate the liquidity ratio as volume divided by market cap.

        Args:
            token (str): The token symbol.

        Returns:
            float: The liquidity ratio.
        """
        try:
            volume = await self.get_token_volume(token)
            metadata = await self.get_token_metadata(token)
            market_cap = metadata.get('market_cap', 0) if metadata else 0
            return volume / market_cap if market_cap > 0 else 0.0
        except Exception as e:
            logger.error(f"Error calculating liquidity ratio: {e}")
            return 0.0

    async def get_token_supply_data(self, token: str) -> Dict[str, Any]:
        """
        Get total and circulating supply for a token.

        Args:
            token (str): The token symbol.

        Returns:
            Dict[str, Any]: A dictionary with 'total_supply' and 'circulating_supply'.
        """
        metadata = await self.get_token_metadata(token)
        if not metadata:
            return {}
        return {
            'total_supply': metadata.get('total_supply', 0),
            'circulating_supply': metadata.get('circulating_supply', 0)
        }

    async def get_token_market_cap(self, token: str) -> float:
        """
        Get the market capitalization for a token.

        Args:
            token (str): The token symbol.

        Returns:
            float: The market cap.
        """
        metadata = await self.get_token_metadata(token)
        return metadata.get('market_cap', 0) if metadata else 0

    async def get_price_change_24h(self, token: str) -> float:
        """
        Get the 24-hour price change percentage for a token.

        Args:
            token (str): The token symbol.

        Returns:
            float: The 24-hour price change percentage.
        """
        metadata = await self.get_token_metadata(token)
        return metadata.get('percent_change_24h', 0) if metadata else 0

    async def update_training_data(self) -> None:
        """
        Update training data for model training by fetching data for all tokens.
        """
        try:
            token_addresses = await self.configuration.get_token_addresses()
            update_tasks = [self._gather_training_data(token) for token in token_addresses]
            updates = await asyncio.gather(*update_tasks, return_exceptions=True)
            valid_updates = [update for update in updates if isinstance(update, dict)]
            if valid_updates:
                await self._write_training_data(valid_updates)
                await self.train_price_model()
        except Exception as e:
            logger.error(f"Error updating training data: {e}")

    async def _write_training_data(self, updates: List[Dict[str, Any]]) -> None:
        """
        Write training data updates to a CSV file.

        Args:
            updates (List[Dict[str, Any]]): List of training data dictionaries.
        """
        try:
            import pandas as pd
            from io import StringIO
            import aiofiles

            df = pd.DataFrame(updates)
            training_data_path = Path(self.configuration.TRAINING_DATA_PATH)
            if training_data_path.exists():
                async with aiofiles.open(training_data_path, 'r') as f:
                    old_data = await f.read()
                if old_data:
                    df_old = pd.read_csv(StringIO(old_data))
                    df = pd.concat([df_old, df], ignore_index=True)
            async with aiofiles.open(training_data_path, 'w', encoding='utf-8') as file:
                await file.write(df.to_csv(index=False))
            await self._cleanup_old_data(training_data_path, days=30)
        except Exception as e:
            logger.error(f"Error writing training data: {e}")

    async def _cleanup_old_data(self, filepath: Path, days: int) -> None:
        """
        Remove training data older than the specified number of days.

        Args:
            filepath (Path): Path to the training data file.
            days (int): Number of days to retain.
        """
        try:
            import pandas as pd
            from io import StringIO
            import aiofiles

            async with aiofiles.open(filepath, 'r') as f:
                content = await f.read()
            if content:
                df = pd.read_csv(StringIO(content))
                cutoff_time = int(time.time()) - (days * 86400)
                df = df[df['timestamp'] >= cutoff_time]
                async with aiofiles.open(filepath, 'w', encoding='utf-8') as file:
                    await file.write(df.to_csv(index=False))
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")

    async def train_price_model(self) -> None:
        """
        Train a linear regression model for price prediction and save the model.
        """
        try:
            training_data_path = Path(self.configuration.TRAINING_DATA_PATH)
            model_path = Path(self.configuration.MODEL_PATH)
            if not training_data_path.exists():
                logger.warning("No training data found")
                return
            df = pd.read_csv(training_data_path)
            if len(df) < self.configuration.MIN_TRAINING_SAMPLES:
                logger.warning(f"Insufficient training data: {len(df)} samples")
                return
            features = ['price_usd', 'volume_24h', 'market_cap', 'volatility',
                        'liquidity_ratio', 'price_momentum']
            X = df[features].fillna(0)
            y = df['price_usd'].fillna(0)
            model = LinearRegression()
            model.fit(X, y)
            joblib.dump(model, model_path)
            logger.debug(f"Price model trained and saved to {model_path}")
        except Exception as e:
            logger.error(f"Error training price model: {e}")

    async def predict_price(self, token: str) -> float:
        """
        Predict the price of a token using the trained model.

        Args:
            token (str): The token symbol.

        Returns:
            float: The predicted price.
        """
        try:
            model_path = Path(self.configuration.MODEL_PATH)
            if not model_path.exists():
                await self.train_price_model()
                if not model_path.exists():
                    return 0.0
            model = joblib.load(model_path)
            market_data = await self._fetch_market_data(token)
            if not market_data:
                return 0.0
            features = pd.DataFrame([market_data])
            prediction = model.predict(features)[0]
            return float(prediction)
        except Exception as e:
            logger.error(f"Error predicting price: {e}")
            return 0.0
