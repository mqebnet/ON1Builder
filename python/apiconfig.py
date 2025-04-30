import asyncio
import os
import aiohttp
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from cachetools import TTLCache, cached
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

# Assuming configuration facade
from configuration import Configuration
from loggingconfig import setup_logging
import logging

logger = setup_logging("APIConfig", level=logging.DEBUG)

# Define constants for timeouts and retries
DEFAULT_REQUEST_TIMEOUT = 10 # seconds
DEFAULT_GATHER_TIMEOUT = 3 # seconds per provider (A12)
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 0.5 # seconds

class APIConfig:
    """
    Manages connections and requests to various external APIs (Exchanges, Price Feeds, Subgraphs)
    for market data, including price, volume, and metadata. Implements caching and rate limiting.
    """
    # A12: Define per-provider timeout for gather
    PROVIDER_TIMEOUT = DEFAULT_GATHER_TIMEOUT

    def __init__(self, configuration: Configuration) -> None:
        self.configuration = configuration
        self.session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock() # Lock for session creation

        # --- API Provider Configurations ---
        # Structure: { 'provider_name': {'base_url': ..., 'endpoints': {...}, 'rate_limit': ..., 'weight': ..., 'api_key_name': ..., 'enabled': ...}}
        self.apiconfigs: Dict[str, Dict[str, Any]] = self._load_api_provider_configs()

        # --- Rate Limiting ---
        # Use asyncio.Semaphore for async rate limiting per provider
        self.rate_limiters: Dict[str, asyncio.Semaphore] = {
            name: asyncio.Semaphore(cfg.get("rate_limit", 1)) # Default limit of 1 req/sec if not specified
            for name, cfg in self.apiconfigs.items() if cfg.get("enabled", True)
        }
        # Track request stats (optional)
        self.request_stats: Dict[str, Dict[str, int]] = {name: {"success": 0, "fail": 0, "limit": 0} for name in self.apiconfigs}

        # --- Caching ---
        # TTLCache for frequently changing data
        self.price_cache: TTLCache = TTLCache(maxsize=2000, ttl=configuration.get_config_value("API_PRICE_CACHE_TTL", 60)) # Short TTL for prices
        self.volume_cache: TTLCache = TTLCache(maxsize=1000, ttl=configuration.get_config_value("API_VOLUME_CACHE_TTL", 300)) # Longer for volume
        self.metadata_cache: TTLCache = TTLCache(maxsize=500, ttl=configuration.get_config_value("API_METADATA_CACHE_TTL", 3600)) # Long for metadata
        self.historical_cache: TTLCache = TTLCache(maxsize=500, ttl=configuration.get_config_value("API_HISTORICAL_CACHE_TTL", 1800)) # Moderate for historical
        # A11 uses predict_price, assuming it's part of APIConfig now
        self.prediction_cache: TTLCache = TTLCache(maxsize=1000, ttl=self.configuration.PREDICTION_CACHE_TTL)


        # --- Token Mapping ---
        self.token_address_to_symbol: Dict[str, str] = {}
        self.token_symbol_to_address: Dict[str, str] = {}
        # Map internal symbol (e.g., ETH) to provider-specific ID (e.g., 'ethereum' for coingecko)
        self.symbol_to_provider_id: Dict[str, Dict[str, str]] = {} # { 'ETH': {'coingecko': 'ethereum', 'coinmarketcap': '1027'}, ... }

        # --- Model Loading (if prediction is handled here) ---
        self._model: Optional[LinearRegression] = None # Placeholder for price prediction model


    def _load_api_provider_configs(self) -> Dict[str, Dict[str, Any]]:
        """Loads API provider details from configuration or defaults."""
        # Example: Load from a dedicated section in Configuration or a separate file
        # Hardcoding for now, replace with dynamic loading
        # A16: Add component name to log extra
        log_extra = {"component": "APIConfig"}
        logger.debug("Loading API provider configurations...", extra=log_extra)

        configs = {
            "binance": {
                "base_url": "https://api.binance.com/api/v3",
                "endpoints": {"price": "/ticker/price", "volume": "/ticker/24hr"},
                "rate_limit": 20, # Per second (adjust based on Binance limits 1200/min)
                "weight": 1.0, # Weight for price aggregation
                "enabled": True,
                "api_key_name": None # Public endpoints assumed
            },
            "coingecko": {
                "base_url": "https://api.coingecko.com/api/v3",
                "endpoints": {
                    "price": "/simple/price",
                    "historical": "/coins/{id}/market_chart", # Requires 'id' parameter
                    "volume": "/coins/{id}/market_chart", # Requires 'id' parameter
                    "metadata": "/coins/{id}"
                 },
                "api_key_name": "COINGECKO_API_KEY", # Matches key in main Configuration
                "rate_limit": 40 if self.configuration.COINGECKO_API_KEY else 8, # Adjust based on free/paid tier
                "weight": 0.9 if self.configuration.COINGECKO_API_KEY else 0.6,
                "enabled": True,
            },
            "coinmarketcap": {
                "base_url": "https://pro-api.coinmarketcap.com/v1",
                 "endpoints": {
                    "price": "/cryptocurrency/quotes/latest", # requires symbol or id
                    "volume": "/cryptocurrency/quotes/latest", # volume is part of quotes
                    "metadata": "/cryptocurrency/info" # requires symbol or id
                 },
                "api_key_name": "COINMARKETCAP_API_KEY",
                "rate_limit": 10, # Adjust based on plan (e.g., 333/day / 86400s)
                "weight": 0.8,
                "enabled": bool(self.configuration.COINMARKETCAP_API_KEY), # Enable only if key exists
            },
             "cryptocompare": {
                "base_url": "https://min-api.cryptocompare.com/data",
                 "endpoints": {
                    "price": "/price", # ?fsym=BTC&tsyms=USD,EUR
                    "historical": "/v2/histoday", # ?fsym=BTC&tsym=USD&limit=30
                    "volume": "/pricemultifull", # ?fsyms=BTC,ETH&tsyms=USD (contains volume)
                 },
                "api_key_name": "CRYPTOCOMPARE_API_KEY",
                "rate_limit": 20, # Adjust based on plan
                "weight": 0.7,
                "enabled": bool(self.configuration.CRYPTOCOMPARE_API_KEY),
            },
            # Add other providers like DexScreener, 1inch, Subgraphs here...
             "dexscreener": {
                "base_url": "https://api.dexscreener.com/latest/dex",
                "endpoints": {
                    "pair_info": "/pairs/{chain}/{pairAddress}", # Provides price, volume etc.
                    "search": "/tokens/search" # Search by symbol/name -> pairAddress
                 },
                 "rate_limit": 5, # Example limit
                 "weight": 0.5,
                 "enabled": True,
                 "api_key_name": None
            },
        }
        # Filter out disabled providers
        enabled_configs = {name: cfg for name, cfg in configs.items() if cfg.get("enabled", True)}
        logger.info("Loaded %d enabled API providers: %s", len(enabled_configs), list(enabled_configs.keys()), extra=log_extra)
        return enabled_configs

    async def _get_session(self) -> aiohttp.ClientSession:
        """Creates or returns the existing aiohttp session."""
        async with self._session_lock:
            if self.session is None or self.session.closed:
                 # Create a new session with default timeout
                 timeout = aiohttp.ClientTimeout(total=DEFAULT_REQUEST_TIMEOUT)
                 self.session = aiohttp.ClientSession(timeout=timeout)
                 logger.debug("Created new aiohttp session.", extra={"component": "APIConfig"})
            return self.session

    async def close(self) -> None:
        """Closes the aiohttp session."""
        log_extra = {"component": "APIConfig"} # A16
        async with self._session_lock:
            if self.session and not self.session.closed:
                await self.session.close()
                self.session = None
                # A2: Parameterized logging
                logger.debug("APIConfig aiohttp session closed.", extra=log_extra)

    async def initialize(self) -> None:
        """Initializes the APIConfig, loading mappings and ensuring session."""
        log_extra = {"component": "APIConfig"} # A16
        logger.info("Initializing APIConfig...", extra=log_extra)
        await self._get_session() # Ensure session exists

        # Load token mappings from configuration files
        await self._load_token_mappings()

        # Load price prediction model if used here
        # await self._load_prediction_model()

        logger.info("APIConfig initialized successfully.", extra=log_extra)

    async def _load_token_mappings(self) -> None:
        """Loads token address/symbol/provider ID mappings from config files."""
        log_extra = {"component": "APIConfig"} # A16
        try:
            # Use Configuration methods to access paths safely
            token_addresses_path = self.configuration.TOKEN_ADDRESSES # Path object
            token_symbols_path = self.configuration.TOKEN_SYMBOLS # Path object

            # Load JSON data using configuration's helper (assuming it's async)
            address_map = await self.configuration._load_json_safe(token_addresses_path, "token addresses")
            symbol_map = await self.configuration._load_json_safe(token_symbols_path, "token symbols provider IDs")

            # Populate internal mappings
            self.token_symbol_to_address.clear()
            self.token_address_to_symbol.clear()
            self.symbol_to_provider_id.clear()

            for symbol, address in address_map.items():
                if symbol and address and not symbol.startswith("_"): # Skip comments/internal keys
                    symbol_upper = symbol.upper()
                    try:
                        # Validate and checksum address
                        checksum_address = self.configuration._validate_ethereum_address(address, f"Token {symbol_upper}")
                        address_lower = checksum_address.lower()

                        self.token_symbol_to_address[symbol_upper] = checksum_address # Store checksummed
                        self.token_address_to_symbol[address_lower] = symbol_upper # Use lowercase for lookup

                        # Add provider IDs if available in the symbol map
                        if symbol_upper in symbol_map:
                             self.symbol_to_provider_id[symbol_upper] = symbol_map[symbol_upper]

                    except ValueError as e:
                        logger.warning("Invalid address for symbol %s in %s: %s. Skipping.", symbol_upper, token_addresses_path.name, e, extra=log_extra)

            logger.info(
                "Loaded %d token symbol->address mappings and %d symbol->provider ID mappings.",
                len(self.token_symbol_to_address), len(self.symbol_to_provider_id), extra=log_extra
            )
            logger.debug("Symbols: %s", list(self.token_symbol_to_address.keys()), extra=log_extra)


        except (FileNotFoundError, ValueError, RuntimeError) as e:
             logger.error("Failed to load token mappings: %s. Token lookups may fail.", e, exc_info=True, extra=log_extra)
             # Continue initialization, but operations needing mappings might fail


    def get_token_symbol(self, address: str) -> Optional[str]:
        """Gets the internal token symbol from a lowercase address."""
        if not address: return None
        return self.token_address_to_symbol.get(address.lower())

    def get_token_address(self, symbol: str) -> Optional[str]:
        """Gets the checksummed token address from an uppercase symbol."""
        if not symbol: return None
        return self.token_symbol_to_address.get(symbol.upper())

    def _get_provider_token_id(self, token_symbol_or_addr: str, provider: str) -> Optional[str]:
        """Gets the provider-specific ID for a given internal token symbol or address."""
        symbol = None
        if token_symbol_or_addr.startswith("0x"):
            symbol = self.get_token_symbol(token_symbol_or_addr)
        else:
            symbol = token_symbol_or_addr.upper()

        if not symbol:
            logger.debug("Cannot find symbol for %s to get provider ID for %s.", token_symbol_or_addr, provider)
            return None # Cannot find symbol

        provider_ids = self.symbol_to_provider_id.get(symbol, {})
        provider_id = provider_ids.get(provider)

        if provider_id:
            return provider_id
        else:
            # Fallback: Use the symbol itself (lowercase for coingecko, uppercase otherwise)
            fallback_id = symbol.lower() if provider == "coingecko" else symbol
            logger.debug("No specific ID for %s on %s. Using fallback ID: %s", symbol, provider, fallback_id)
            return fallback_id


    async def _make_request(
        self,
        provider: str,
        endpoint_key: str, # e.g., "price", "historical"
        params: Optional[Dict[str, Any]] = None,
        path_params: Optional[Dict[str, str]] = None, # For URLs like /coins/{id}/...
        method: str = "GET", # Typically GET, sometimes POST (e.g., subgraphs)
        data: Optional[Any] = None # For POST requests
    ) -> Optional[Any]:
        """Internal helper to make rate-limited, retried requests to a provider."""
        log_extra = {"component": "APIConfig", "provider": provider, "endpoint": endpoint_key} # A16
        cfg = self.apiconfigs.get(provider)
        if not cfg or not cfg.get("enabled", True):
            logger.warning("Attempted request to disabled/unknown provider: %s", provider, extra=log_extra)
            return None

        limiter = self.rate_limiters.get(provider)
        if not limiter:
            logger.error("Rate limiter not found for provider: %s", provider, extra=log_extra)
            return None # Should not happen if setup is correct

        endpoint_path_template = cfg.get("endpoints", {}).get(endpoint_key)
        if not endpoint_path_template:
            logger.error("Endpoint key '%s' not configured for provider %s.", endpoint_key, provider, extra=log_extra)
            return None

        # Format URL path with parameters
        try:
            url_path = endpoint_path_template.format(**(path_params or {}))
        except KeyError as e:
             logger.error("Missing path parameter '%s' for provider %s, endpoint %s.", e, provider, endpoint_key, extra=log_extra)
             return None

        full_url = cfg["base_url"] + url_path

        # Prepare headers (e.g., API key)
        headers = {"Accept": "application/json"}
        api_key_name = cfg.get("api_key_name")
        if api_key_name:
            api_key = getattr(self.configuration, api_key_name, None)
            if api_key:
                # Determine header name based on provider conventions
                if provider == "coinmarketcap":
                    headers["X-CMC_PRO_API_KEY"] = api_key
                elif provider == "coingecko" and self.configuration.COINGECKO_API_KEY: # Only add if pro key exists
                     headers["x-cg-pro-api-key"] = api_key
                elif provider == "cryptocompare":
                     # Key often added as URL param, but header might be supported
                     headers["authorization"] = f"Apikey {api_key}"
                     # Also add to params for safety, some endpoints require it there
                     if params is None: params = {}
                     params["api_key"] = api_key
                else:
                     logger.warning("API key found for %s, but header format unknown.", provider, extra=log_extra)
            else:
                logger.debug("API key name '%s' configured for %s, but key not found in main config.", api_key_name, provider, extra=log_extra)
                # Proceed without key if endpoint allows


        session = await self._get_session()
        current_retry_delay = self.retry_delay

        # Rate limiting and retry loop
        async with limiter: # Acquire semaphore permit for this provider
            for attempt in range(self.max_retries):
                log_extra_attempt = {**log_extra, "attempt": attempt + 1}
                try:
                    # A2: Parameterized logging
                    logger.debug("Making request to %s: %s %s (Params: %s)", provider, method, full_url, params, extra=log_extra_attempt)
                    async with session.request(method, full_url, params=params, headers=headers, json=data) as resp: # Use json for POST

                        # Check for rate limiting first
                        if resp.status == 429:
                             self.request_stats[provider]["limit"] += 1
                             # A2: Parameterized logging
                             logger.warning("Rate limit exceeded (429) for %s (attempt %d/%d). Retrying after delay.",
                                          provider, attempt + 1, self.max_retries, extra=log_extra_attempt)
                             # Use 'Retry-After' header if available, otherwise backoff
                             retry_after = int(resp.headers.get("Retry-After", str(int(current_retry_delay))))
                             await asyncio.sleep(retry_after)
                             current_retry_delay *= 1.5 # Increase backoff for next time if header missing
                             continue # Go to next attempt

                        # Check for other client/server errors
                        resp.raise_for_status() # Raises ClientResponseError for 4xx/5xx

                        # Success
                        response_data = await resp.json()
                        self.request_stats[provider]["success"] += 1
                        # A2: Parameterized logging
                        logger.debug("Request to %s successful (Status %d).", provider, resp.status, extra=log_extra_attempt)
                        return response_data

                except aiohttp.ClientResponseError as e:
                    self.request_stats[provider]["fail"] += 1
                    # A2: Parameterized logging
                    logger.warning(
                        "Request to %s failed (Status %d, Attempt %d/%d): %s. URL: %s",
                        provider, e.status, attempt + 1, self.max_retries, e.message, full_url, extra=log_extra_attempt
                    )
                    # Specific handling for common errors? (e.g., 400 Bad Request, 401 Unauthorized, 404 Not Found)
                    if e.status in [400, 401, 403, 404]:
                         logger.error("Unrecoverable client error (%d) for %s. Aborting retries.", e.status, provider, extra=log_extra_attempt)
                         return None # Don't retry these errors
                    # Retry other errors (5xx, maybe timeouts handled below)

                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                     # Includes connection errors, timeouts, etc.
                    self.request_stats[provider]["fail"] += 1
                    # A2: Parameterized logging
                    logger.warning(
                        "Request to %s failed (Attempt %d/%d): %s (%s). Retrying...",
                         provider, attempt + 1, self.max_retries, type(e).__name__, e, extra=log_extra_attempt
                    )

                except Exception as e:
                    # Catch unexpected errors during request/response processing
                    self.request_stats[provider]["fail"] += 1
                    # A2: Parameterized logging
                    logger.error(
                        "Unexpected error during request to %s (Attempt %d/%d): %s",
                        provider, attempt + 1, self.max_retries, e, exc_info=True, extra=log_extra_attempt
                    )
                    # Potentially break retry loop for unexpected errors


                # Wait before retrying (if not the last attempt)
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(current_retry_delay)
                    current_retry_delay *= 1.5 # Exponential backoff

        # A2: Parameterized logging
        logger.error("All %d request attempts failed for %s endpoint '%s'. URL: %s",
                     self.max_retries, provider, endpoint_key, full_url, extra=log_extra)
        return None


    # A12: Refactor get_real_time_price with asyncio.gather and timeout
    async def get_real_time_price(self, token_or_addr: str, vs: str = "usd") -> Optional[Decimal]:
        """
        Fetches the real-time price of a token from multiple providers concurrently
        and returns a weighted average.

        Args:
            token_or_addr: Token symbol (e.g., "ETH") or address ("0x...").
            vs: The currency to get the price against (e.g., "usd", "eth").

        Returns:
            The weighted average price as a Decimal, or None if all providers fail.
        """
        log_extra = {"component": "APIConfig", "token": token_or_addr, "vs_currency": vs} # A16
        vs_upper = vs.upper()

        # 1. Normalize token input to internal symbol
        symbol = None
        if token_or_addr.startswith("0x"):
            symbol = self.get_token_symbol(token_or_addr)
        else:
            symbol = token_or_addr.upper()

        if not symbol:
            logger.warning("Cannot find symbol for '%s'.", token_or_addr, extra=log_extra)
            return None
        log_extra["symbol"] = symbol # Add symbol to context

        # 2. Check cache
        cache_key = f"price:{symbol}:{vs_upper}"
        if cache_key in self.price_cache:
            # A2: Parameterized logging
            logger.debug("Price cache hit for %s/%s.", symbol, vs_upper, extra=log_extra)
            return self.price_cache[cache_key]

        # 3. Prepare concurrent tasks for enabled providers
        tasks = []
        providers_involved = []
        for provider, cfg in self.apiconfigs.items():
            if cfg.get("enabled", True):
                # Create a task for each provider's fetch function
                # Add timeout wrapper around each individual provider task
                task = asyncio.create_task(
                    asyncio.wait_for(
                        self._fetch_single_provider_price(provider, symbol, vs_upper, log_extra),
                        timeout=self.PROVIDER_TIMEOUT
                    ),
                    name=f"PriceFetch-{provider}-{symbol}"
                )
                tasks.append(task)
                providers_involved.append(provider)

        if not tasks:
            logger.warning("No enabled API providers found for price fetching.", extra=log_extra)
            return None

        # A2: Parameterized logging
        logger.debug("Fetching price for %s/%s from %d providers concurrently...", symbol, vs_upper, len(tasks), extra=log_extra)

        # 4. Run tasks concurrently with asyncio.gather
        # return_exceptions=True prevents gather from stopping on the first error/timeout
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 5. Process results and calculate weighted average
        prices: List[Decimal] = []
        weights: List[float] = []
        total_weight = 0.0

        for result, provider in zip(results, providers_involved):
            log_extra_provider = {**log_extra, "provider": provider}
            price_decimal: Optional[Decimal] = None

            if isinstance(result, asyncio.TimeoutError):
                 # A2: Parameterized logging
                 logger.warning("Price fetch timed out for provider %s.", provider, extra=log_extra_provider)
                 self.request_stats[provider]["fail"] += 1 # Count timeout as failure
            elif isinstance(result, Exception):
                 # A2: Parameterized logging
                 logger.warning("Price fetch failed for provider %s: %s", provider, result, extra=log_extra_provider)
                 # Already counted as fail in _make_request if it got that far
            elif isinstance(result, Decimal):
                 price_decimal = result
                 # A2: Parameterized logging
                 logger.debug("Received price %.6f from %s.", price_decimal, provider, extra=log_extra_provider)
            else:
                # Result is None or unexpected type
                logger.debug("Provider %s returned no valid price.", provider, extra=log_extra_provider)


            # If a valid price was obtained, add it to the list for averaging
            if price_decimal is not None and price_decimal > 0:
                 provider_weight = self.apiconfigs[provider].get("weight", 0.1) # Get configured weight
                 prices.append(price_decimal)
                 weights.append(provider_weight)
                 total_weight += provider_weight

        # 6. Calculate weighted average if prices were found
        if prices and total_weight > 0:
            weighted_sum = sum(p * Decimal(str(w)) for p, w in zip(prices, weights))
            final_price = weighted_sum / Decimal(str(total_weight))
            # A2: Parameterized logging + A16
            logger.info(
                "Calculated weighted average price for %s/%s: %.8f (from %d sources)",
                symbol, vs_upper, final_price, len(prices), extra=log_extra
            )
            # Cache the result
            self.price_cache[cache_key] = final_price
            return final_price
        else:
            # A2: Parameterized logging + A16
            logger.warning("Failed to get valid price for %s/%s from any provider.", symbol, vs_upper, extra=log_extra)
            # Cache failure briefly? (e.g., cache None for a few seconds)
            # self.price_cache[cache_key] = None # Be careful caching None if TTL is long
            return None


    async def _fetch_single_provider_price(self, provider: str, symbol: str, vs_currency: str, parent_log_extra: Dict) -> Optional[Decimal]:
        """Fetches price from a single provider, handling specific endpoint logic."""
        log_extra = {**parent_log_extra, "provider": provider} # Add provider to context
        provider_token_id = self._get_provider_token_id(symbol, provider)
        if not provider_token_id:
             logger.warning("No token ID found for %s on %s.", symbol, provider, extra=log_extra)
             return None

        params: Dict[str, Any] = {}
        path_params: Dict[str, str] = {}
        endpoint_key: str = "price"

        try:
            # Provider-specific logic for price endpoints
            if provider == "binance":
                # Binance uses symbol pairs like 'BTCUSDT'
                # Handle ETH vs WETH, stablecoin pairs carefully
                vs_map = {"USD": "USDT", "ETH": "ETH"} # Simple mapping
                if vs_currency not in vs_map: return None # Unsupported vs currency for this mapping
                binance_symbol = f"{provider_token_id}{vs_map[vs_currency]}"
                params = {"symbol": binance_symbol}
                data = await self._make_request(provider, endpoint_key, params=params)
                return Decimal(data["price"]) if data and "price" in data else None

            elif provider == "coingecko":
                # Uses 'ids' and 'vs_currencies'
                params = {"ids": provider_token_id, "vs_currencies": vs_currency.lower()}
                data = await self._make_request(provider, endpoint_key, params=params)
                # Response format: {'bitcoin': {'usd': 29000.5}}
                price_data = data.get(provider_token_id, {}) if data else {}
                return Decimal(str(price_data[vs_currency.lower()])) if vs_currency.lower() in price_data else None

            elif provider == "coinmarketcap":
                 params = {"symbol": provider_token_id, "convert": vs_currency.upper()}
                 data = await self._make_request(provider, endpoint_key, params=params)
                 # Response: {'data': {'BTC': {'quote': {'USD': {'price': ...}}}}}
                 quote = data.get("data", {}).get(provider_token_id.upper(), {}).get("quote", {}).get(vs_currency.upper(), {}) if data else {}
                 return Decimal(str(quote["price"])) if quote and "price" in quote else None

            elif provider == "cryptocompare":
                 params = {"fsym": provider_token_id.upper(), "tsyms": vs_currency.upper()}
                 data = await self._make_request(provider, endpoint_key, params=params)
                 # Response: {'USD': 29000.5}
                 return Decimal(str(data[vs_currency.upper()])) if data and vs_currency.upper() in data else None

            elif provider == "dexscreener":
                 # DexScreener requires pair address or search; more complex
                 # Assuming we have pair address mapping or use search endpoint
                 # Placeholder: needs pair address lookup logic
                 logger.debug("DexScreener price fetching needs pair address logic.", extra=log_extra)
                 # Example using search (less reliable):
                 # search_params = {"q": provider_token_id}
                 # search_data = await self._make_request(provider, "search", params=search_params)
                 # if search_data and search_data.get("pairs"):
                 #     # Find best pair (e.g., highest liquidity with WETH or USDC)
                 #     # pair_addr = find_best_pair(search_data["pairs"])
                 #     pair_addr = search_data["pairs"][0].get("pairAddress") # Simplistic: take first pair
                 #     if pair_addr:
                 #          chain = "ethereum" # Assume eth
                 #          pair_data = await self._make_request(provider, "pair_info", path_params={"chain": chain, "pairAddress": pair_addr})
                 #          price_str = pair_data.get("pair", {}).get("priceUsd") if pair_data else None
                 #          # Convert USD price if vs_currency is not USD (e.g., use ETH price)
                 #          if price_str and vs_currency.upper() == "USD":
                 #                return Decimal(price_str)
                 return None # Requires more implementation

            else:
                logger.warning("Price fetching logic not implemented for provider: %s", provider, extra=log_extra)
                return None

        except (InvalidOperation, TypeError, KeyError, IndexError) as e:
             logger.warning("Error parsing price response from %s for %s: %s", provider, symbol, e, extra=log_extra)
             return None
        except Exception as e:
             # Catch errors from _make_request or parsing
             logger.error("Unexpected error fetching price from %s for %s: %s", provider, symbol, e, exc_info=True, extra=log_extra)
             return None


    # --- Other Data Fetching Methods (Volume, Historical, Metadata) ---
    # These should also be refactored similarly to use _make_request and handle provider specifics

    async def get_token_volume(self, token_or_addr: str, vs: str = "usd") -> Optional[float]:
        """Fetches 24h trading volume for the token."""
        log_extra = {"component": "APIConfig", "token": token_or_addr, "vs_currency": vs}
        symbol = self._resolve_symbol(token_or_addr, log_extra)
        if not symbol: return None
        vs_upper = vs.upper()

        cache_key = f"volume:{symbol}:{vs_upper}"
        if cache_key in self.volume_cache:
            logger.debug("Volume cache hit for %s/%s.", symbol, vs_upper, extra=log_extra)
            return self.volume_cache[cache_key]

        # Prioritize providers known for good volume data
        provider_preference = ["binance", "coinmarketcap", "coingecko", "cryptocompare", "dexscreener"]
        for provider in provider_preference:
            if provider in self.apiconfigs and self.apiconfigs[provider].get("enabled"):
                 volume = await self._fetch_single_provider_volume(provider, symbol, vs_upper, log_extra)
                 if volume is not None:
                      self.volume_cache[cache_key] = volume
                      logger.info("Fetched volume for %s/%s from %s: %.2f", symbol, vs_upper, provider, volume, extra=log_extra)
                      return volume

        logger.warning("Failed to get volume for %s/%s from any provider.", symbol, vs_upper, extra=log_extra)
        return None # Return None if no provider succeeds

    async def _fetch_single_provider_volume(self, provider: str, symbol: str, vs_currency: str, parent_log_extra: dict) -> Optional[float]:
        """Fetches volume from a single provider."""
        log_extra = {**parent_log_extra, "provider": provider}
        provider_token_id = self._get_provider_token_id(symbol, provider)
        if not provider_token_id: return None

        params: Dict[str, Any] = {}
        path_params: Dict[str, str] = {}
        endpoint_key: str = "volume" # Key for volume endpoint in config

        try:
             if provider == "binance":
                  # Assumes USDT pair for USD volume
                  if vs_currency != "USD": return None # Only support USD via USDT for now
                  binance_symbol = f"{provider_token_id}USDT"
                  params = {"symbol": binance_symbol}
                  data = await self._make_request(provider, endpoint_key, params=params)
                  # 'quoteVolume' is volume in the quote asset (USDT)
                  return float(data["quoteVolume"]) if data and "quoteVolume" in data else None

             elif provider == "coingecko":
                  # Uses market_chart endpoint, requires paid API for volume usually
                  if not self.configuration.COINGECKO_API_KEY: return None # Free tier might lack volume
                  path_params = {"id": provider_token_id}
                  params = {"vs_currency": vs_currency.lower(), "days": "1"} # Get last day's volume
                  data = await self._make_request(provider, endpoint_key, params=params, path_params=path_params)
                  # Response: {'total_volumes': [[timestamp, volume], ...]}
                  volumes = data.get("total_volumes", []) if data else []
                  return float(volumes[-1][1]) if volumes else None

             elif provider == "coinmarketcap":
                  params = {"symbol": provider_token_id, "convert": vs_currency.upper()}
                  data = await self._make_request(provider, endpoint_key, params=params) # Uses 'price' endpoint key which contains volume
                  quote = data.get("data", {}).get(provider_token_id.upper(), {}).get("quote", {}).get(vs_currency.upper(), {}) if data else {}
                  # Key might be 'volume_24h' or similar
                  return float(quote["volume_24h"]) if quote and "volume_24h" in quote else None

             elif provider == "cryptocompare":
                  # pricemultifull provides volume data
                  params = {"fsyms": provider_token_id.upper(), "tsyms": vs_currency.upper()}
                  data = await self._make_request(provider, "volume", params=params) # Use specific 'volume' endpoint key
                  # Response: {'RAW': {'BTC': {'USD': {'VOLUME24HOURTO': ...}}}}
                  raw_data = data.get("RAW", {}).get(provider_token_id.upper(), {}).get(vs_currency.upper(), {}) if data else {}
                  return float(raw_data["VOLUME24HOURTO"]) if raw_data and "VOLUME24HOURTO" in raw_data else None

             # Add DexScreener volume logic if needed (usually part of pair_info)

             else:
                  logger.warning("Volume fetching not implemented for provider: %s", provider, extra=log_extra)
                  return None

        except (ValueError, TypeError, KeyError, IndexError) as e:
             logger.warning("Error parsing volume response from %s for %s: %s", provider, symbol, e, extra=log_extra)
             return None
        except Exception as e:
             logger.error("Unexpected error fetching volume from %s for %s: %s", provider, symbol, e, exc_info=True, extra=log_extra)
             return None


    async def get_token_metadata(self, token_or_addr: str) -> Optional[Dict[str, Any]]:
        """Fetches metadata (market cap, supply, etc.) for the token."""
        log_extra = {"component": "APIConfig", "token": token_or_addr}
        symbol = self._resolve_symbol(token_or_addr, log_extra)
        if not symbol: return None

        cache_key = f"metadata:{symbol}"
        if cache_key in self.metadata_cache:
            logger.debug("Metadata cache hit for %s.", symbol, extra=log_extra)
            return self.metadata_cache[cache_key]

        provider_preference = ["coinmarketcap", "coingecko"] # Providers good for metadata
        for provider in provider_preference:
             if provider in self.apiconfigs and self.apiconfigs[provider].get("enabled"):
                  metadata = await self._fetch_single_provider_metadata(provider, symbol, log_extra)
                  if metadata:
                       self.metadata_cache[cache_key] = metadata
                       logger.info("Fetched metadata for %s from %s.", symbol, provider, extra=log_extra)
                       return metadata

        logger.warning("Failed to get metadata for %s from any provider.", symbol, extra=log_extra)
        return None

    async def _fetch_single_provider_metadata(self, provider: str, symbol: str, parent_log_extra: dict) -> Optional[Dict[str, Any]]:
         """Fetches metadata from a single provider."""
         log_extra = {**parent_log_extra, "provider": provider}
         provider_token_id = self._get_provider_token_id(symbol, provider)
         if not provider_token_id: return None

         params: Dict[str, Any] = {}
         path_params: Dict[str, str] = {}
         endpoint_key: str = "metadata"

         try:
              if provider == "coingecko":
                   path_params = {"id": provider_token_id}
                   # Params to reduce response size? e.g., localization=false, tickers=false, etc.
                   params = {"localization": "false", "tickers": "false", "community_data": "false", "developer_data": "false", "sparkline": "false"}
                   data = await self._make_request(provider, endpoint_key, params=params, path_params=path_params)
                   if not data: return None
                   # Extract relevant fields
                   market_data = data.get("market_data", {})
                   return {
                       "symbol": data.get("symbol", "").upper(),
                       "name": data.get("name"),
                       "market_cap": market_data.get("market_cap", {}).get("usd"),
                       "total_volume": market_data.get("total_volume", {}).get("usd"),
                       "total_supply": market_data.get("total_supply"),
                       "circulating_supply": market_data.get("circulating_supply"),
                       "description": data.get("description", {}).get("en"),
                       # Add more fields as needed (links, platforms etc)
                   }

              elif provider == "coinmarketcap":
                   params = {"symbol": provider_token_id}
                   data = await self._make_request(provider, endpoint_key, params=params)
                   if not data or "data" not in data or provider_token_id.upper() not in data["data"]: return None
                   info = data["data"][provider_token_id.upper()]
                   # Also fetch quotes for market cap/supply? Requires another call or different endpoint
                   quote_params = {"symbol": provider_token_id, "convert": "USD"}
                   quote_data = await self._make_request(provider, "price", params=quote_params) # Use price endpoint for quotes
                   quote = quote_data.get("data", {}).get(provider_token_id.upper(), {}).get("quote", {}).get("USD", {}) if quote_data else {}

                   return {
                        "symbol": info.get("symbol"),
                        "name": info.get("name"),
                        "market_cap": quote.get("market_cap"),
                        "total_supply": quote.get("total_supply"),
                        "circulating_supply": quote.get("circulating_supply"),
                        "description": info.get("description"),
                        "urls": info.get("urls"),
                        "platform": info.get("platform"), # e.g., Ethereum address
                        "tags": info.get("tags"),
                   }

              else:
                  logger.warning("Metadata fetching not implemented for provider: %s", provider, extra=log_extra)
                  return None

         except (ValueError, TypeError, KeyError, IndexError) as e:
              logger.warning("Error parsing metadata response from %s for %s: %s", provider, symbol, e, extra=log_extra)
              return None
         except Exception as e:
              logger.error("Unexpected error fetching metadata from %s for %s: %s", provider, symbol, e, exc_info=True, extra=log_extra)
              return None


    async def get_token_price_data(
        self, token_or_addr: str, data_type: str = "current", timeframe: int = 1, vs: str = "usd"
    ) -> Union[Optional[Decimal], List[float], None]: # Return type hint refined
        """
        Public method to get current price, historical prices, or other data types.
        Routes to specific methods like get_real_time_price or fetch_historical_prices.
        """
        log_extra = {"component": "APIConfig", "token": token_or_addr, "data_type": data_type} # A16
        symbol = self._resolve_symbol(token_or_addr, log_extra)
        if not symbol: return None

        vs_upper = vs.upper()
        cache_key = f"{data_type}:{symbol}:{vs_upper}:{timeframe}" # Include timeframe in key

        # Check cache based on data type
        target_cache: TTLCache
        if data_type == "current":
             target_cache = self.price_cache
        elif data_type == "historical":
             target_cache = self.historical_cache
        # Add other caches if needed (e.g., volume_cache)
        else:
             target_cache = self.price_cache # Default?

        if cache_key in target_cache:
              logger.debug("Cache hit for %s data (%s).", data_type, cache_key, extra=log_extra)
              return target_cache[cache_key]

        # Fetch data based on type
        result: Union[Optional[Decimal], List[float], None] = None
        if data_type == "current":
            result = await self.get_real_time_price(symbol, vs=vs_upper)
        elif data_type == "historical":
            result = await self.fetch_historical_prices(symbol, days=timeframe, vs=vs_upper)
        elif data_type == "volume":
             volume_float = await self.get_token_volume(symbol, vs=vs_upper)
             result = Decimal(str(volume_float)) if volume_float is not None else None # Convert volume to Decimal for consistency? Or keep float? Let's return float for volume.
             # Adjust return type hint if volume returns float
        elif data_type == "metadata":
             result = await self.get_token_metadata(symbol) # Returns dict
             # Adjust return type hint if metadata returns dict
        else:
            logger.error("Unsupported data_type requested: %s", data_type, extra=log_extra)
            raise ValueError(f"Unsupported data_type: {data_type}")

        # Cache the result if successful
        if result is not None:
            target_cache[cache_key] = result
            logger.debug("Fetched and cached %s data for %s.", data_type, symbol, extra=log_extra)
        else:
             logger.debug("Failed to fetch %s data for %s.", data_type, symbol, extra=log_extra)


        # Ensure return type matches annotation (might need adjustment based on data_type)
        if data_type == "current":
             return result if isinstance(result, Decimal) or result is None else None
        elif data_type == "historical":
             return result if isinstance(result, list) else [] # Return empty list on failure
        # Handle volume/metadata return types if needed
        else:
             return result # Return whatever was fetched


    async def fetch_historical_prices(self, token_or_addr: str, days: int = 30, vs: str = "usd") -> List[float]:
        """Fetches historical price data points."""
        log_extra = {"component": "APIConfig", "token": token_or_addr, "days": days, "vs": vs}
        symbol = self._resolve_symbol(token_or_addr, log_extra)
        if not symbol: return []
        vs_lower = vs.lower()

        cache_key = f"historical:{symbol}:{vs_lower}:{days}"
        if cache_key in self.historical_cache:
            logger.debug("Historical price cache hit for %s/%s (%d days).", symbol, vs_lower, days, extra=log_extra)
            return self.historical_cache[cache_key]

        # Try providers
        provider_preference = ["coingecko", "cryptocompare"] # Good for historical
        for provider in provider_preference:
             if provider in self.apiconfigs and self.apiconfigs[provider].get("enabled"):
                  prices = await self._fetch_single_provider_historical(provider, symbol, days, vs_lower, log_extra)
                  if prices:
                       self.historical_cache[cache_key] = prices
                       logger.info("Fetched %d historical prices for %s/%s from %s.", len(prices), symbol, vs_lower, provider, extra=log_extra)
                       return prices

        logger.warning("Failed to get historical prices for %s/%s from any provider.", symbol, vs_lower, extra=log_extra)
        return []


    async def _fetch_single_provider_historical(self, provider: str, symbol: str, days: int, vs_currency: str, parent_log_extra: dict) -> Optional[List[float]]:
        """Fetches historical prices from a single provider."""
        log_extra = {**parent_log_extra, "provider": provider}
        provider_token_id = self._get_provider_token_id(symbol, provider)
        if not provider_token_id: return None

        params: Dict[str, Any] = {}
        path_params: Dict[str, str] = {}
        endpoint_key: str = "historical"

        try:
            if provider == "coingecko":
                 path_params = {"id": provider_token_id}
                 params = {"vs_currency": vs_currency, "days": str(days)}
                 data = await self._make_request(provider, endpoint_key, params=params, path_params=path_params)
                 # Response: {'prices': [[timestamp, price], ...]}
                 prices_data = data.get("prices", []) if data else []
                 return [float(p[1]) for p in prices_data] if prices_data else None

            elif provider == "cryptocompare":
                 params = {"fsym": symbol.upper(), "tsym": vs_currency.upper(), "limit": str(days - 1)} # limit is N-1 for N days
                 data = await self._make_request(provider, endpoint_key, params=params)
                 # Response: {'Data': {'Data': [{'time': ..., 'close': ...}, ...]}}
                 price_entries = data.get("Data", {}).get("Data", []) if data and data.get("Response") == "Success" else []
                 return [float(entry["close"]) for entry in price_entries] if price_entries else None

            else:
                 logger.warning("Historical price fetching not implemented for provider: %s", provider, extra=log_extra)
                 return None

        except (ValueError, TypeError, KeyError, IndexError) as e:
             logger.warning("Error parsing historical response from %s for %s: %s", provider, symbol, e, extra=log_extra)
             return None
        except Exception as e:
             logger.error("Unexpected error fetching historical from %s for %s: %s", provider, symbol, e, exc_info=True, extra=log_extra)
             return None


    # --- Prediction Methods (Placeholder) ---
    # A11 requires predict_price
    async def _load_prediction_model(self) -> None:
        """Loads the price prediction model if path is configured."""
        log_extra = {"component": "APIConfig"}
        model_path_str = self.configuration.get_config_value("MODEL_PATH")
        if not model_path_str:
             logger.info("MODEL_PATH not configured. Price prediction disabled.", extra=log_extra)
             return
        model_path = Path(model_path_str)
        if await asyncio.to_thread(model_path.exists):
             try:
                  self._model = await asyncio.to_thread(joblib.load, model_path)
                  logger.info("Price prediction model loaded from %s.", model_path, extra=log_extra)
             except Exception as e:
                  logger.error("Failed to load prediction model from %s: %s", model_path, e, extra=log_extra)
        else:
             logger.warning("Prediction model file not found at %s.", model_path, extra=log_extra)

    async def train_price_model(self) -> None:
         """Placeholder for model training logic if managed by APIConfig."""
         # This logic might belong in MarketMonitor instead.
         logger.warning("train_price_model called in APIConfig, but logic likely belongs elsewhere (e.g., MarketMonitor).")
         # If training happens here, load data, fit model, save model.
         pass

    async def predict_price(self, token_or_addr: str) -> Optional[float]:
        """Predicts the price using the loaded model."""
        # This logic might belong in MarketMonitor.
        log_extra = {"component": "APIConfig", "token": token_or_addr}
        symbol = self._resolve_symbol(token_or_addr, log_extra)
        if not symbol: return None

        if self._model is None:
            logger.debug("Prediction model not loaded. Cannot predict.", extra=log_extra)
            return None

        cache_key = f"predict:{symbol}"
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]

        # Requires fetching features similar to MarketMonitor._prepare_prediction_features
        logger.warning("predict_price called in APIConfig, requires feature preparation logic.", extra=log_extra)
        # features_df = await self._prepare_prediction_features_api(symbol, log_extra) # Needs implementation
        # if features_df is None: return None
        # prediction = await asyncio.to_thread(self._model.predict, features_df)
        # result = float(prediction[0])
        result = None # Placeholder
        if result is not None:
            self.prediction_cache[cache_key] = result
        return result


    def _resolve_symbol(self, token_or_addr: str, log_extra: Dict) -> Optional[str]:
        """Helper to consistently get the uppercase symbol from address or symbol."""
        symbol = None
        if token_or_addr.startswith("0x"):
            symbol = self.get_token_symbol(token_or_addr)
        else:
            symbol = token_or_addr.upper()
        if not symbol:
             logger.warning("Cannot resolve symbol for '%s'.", token_or_addr, extra=log_extra)
        return symbol

