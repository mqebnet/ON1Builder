from pathlib import Path
from typing import Callable, Optional, Dict, Any
import logging

logger = logging.getLogger("ConfigPaths") # Use child logger

class ConfigPathResolver:
    """Handles resolution and loading of file paths and contract addresses."""

    # Define default relative paths (relative to project base path)
    DEFAULT_PATHS = {
        "ABI_DIR": "abi",
        "ERC20_ABI_FILE": "erc20_abi.json",
        "AAVE_FLASHLOAN_ABI_FILE": "aave_flashloan_abi.json",
        "AAVE_POOL_ABI_FILE": "aave_pool_abi.json",
        "UNISWAP_ABI_FILE": "uniswap_abi.json",
        "SUSHISWAP_ABI_FILE": "sushiswap_abi.json",
        "GAS_PRICE_ORACLE_ABI_FILE": "gas_price_oracle_abi.json", # If used
        "UTILS_DIR": "utils", # Assuming utils contains JSON configs
        "TOKEN_ADDRESSES_FILE": "token_addresses.json",
        "TOKEN_SYMBOLS_FILE": "token_symbols.json",
        "ERC20_SIGNATURES_FILE": "erc20_signatures.json", # If used
        # Model/Data paths defined in core based on runtime dir
    }

    # Default contract addresses (Mainnet examples - Should be overridden by env)
    DEFAULT_ADDRESSES = {
        "WETH_ADDRESS": "0xC02aaa39b223FE8D0a0e5C4F27eAD9083C756Cc2",
        "USDC_ADDRESS": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        "USDT_ADDRESS": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
        "UNISWAP_ADDRESS": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D", # V2 Router
        "SUSHISWAP_ADDRESS": "0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F", # Router
        "AAVE_POOL_ADDRESS": "0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2", # Aave V3 Pool Ethereum
        "AAVE_ADDRESS_PROVIDER_ADDRESS": "0xa97684ead0e402dC232d5A977953DF7ECBaB3CDb", # Aave V3 Provider Ethereum
        # Flashloan related addresses depend on chosen provider/receiver
        # Example: Aave V3 Flashloan simple receiver (deploy your own)
        "AAVE_FLASHLOAN_ADDRESS": None, # Must be configured via env
        "GAS_PRICE_ORACLE_ADDRESS": None, # Chainlink or other oracle
    }

    def __init__(self, base_path: Path, get_env_str_func: Callable):
        """
        Initializes the resolver.

        Args:
            base_path: The project's base directory path.
            get_env_str_func: A function (like _get_env_str from ConfigCore)
                              to retrieve environment variables.
        """
        self._base_path = base_path
        self._get_env_str = get_env_str_func
        self._validate_address_func = self._get_default_address_validator() # Internal validator
        logger.debug("ConfigPathResolver initialized. Base path: %s", self._base_path)

    def _get_default_address_validator(self) -> Callable:
         """ Provides a basic address validator if needed internally. """
         def validate(address: Optional[str], var_name: str) -> Optional[str]:
              if address is None: return None
              if not isinstance(address, str) or not address.startswith("0x") or len(address) != 42:
                   raise ValueError(f"Invalid format for {var_name}: '{address}'")
              # Basic check, full validation might happen in ConfigCore
              return address
         return validate

    def load_paths(self) -> None:
        """Loads and resolves all relevant file paths and contract addresses."""
        logger.debug("Loading configuration paths and addresses...")

        # --- Resolve Directories ---
        self.ABI_DIR = self._resolve_path("ABI_DIR", self.DEFAULT_PATHS["ABI_DIR"], is_dir=True)
        self.UTILS_DIR = self._resolve_path("UTILS_DIR", self.DEFAULT_PATHS["UTILS_DIR"], is_dir=True)

        # --- Resolve ABI File Paths ---
        # Example: ERC20 ABI
        abi_file_name = self._get_env_str("ERC20_ABI_FILE", self.DEFAULT_PATHS["ERC20_ABI_FILE"], "ERC20 ABI filename")
        self.ERC20_ABI = self._resolve_path_in_dir(self.ABI_DIR, abi_file_name, "ERC20 ABI", required=True)
        # Repeat for other ABIs...
        self.AAVE_FLASHLOAN_ABI = self._resolve_path_in_dir(
            self.ABI_DIR,
            self._get_env_str("AAVE_FLASHLOAN_ABI_FILE", self.DEFAULT_PATHS["AAVE_FLASHLOAN_ABI_FILE"], "Aave Flashloan ABI filename"),
            "Aave Flashloan ABI", required=True
        )
        self.AAVE_POOL_ABI = self._resolve_path_in_dir(
             self.ABI_DIR,
             self._get_env_str("AAVE_POOL_ABI_FILE", self.DEFAULT_PATHS["AAVE_POOL_ABI_FILE"], "Aave Pool ABI filename"),
             "Aave Pool ABI", required=True
         )
        self.UNISWAP_ABI = self._resolve_path_in_dir(
             self.ABI_DIR,
             self._get_env_str("UNISWAP_ABI_FILE", self.DEFAULT_PATHS["UNISWAP_ABI_FILE"], "Uniswap ABI filename"),
             "Uniswap ABI", required=True
         )
        self.SUSHISWAP_ABI = self._resolve_path_in_dir(
             self.ABI_DIR,
             self._get_env_str("SUSHISWAP_ABI_FILE", self.DEFAULT_PATHS["SUSHISWAP_ABI_FILE"], "Sushiswap ABI filename"),
             "Sushiswap ABI", required=True
         )
        # Optional Gas Price Oracle ABI
        gas_oracle_file = self._get_env_str("GAS_PRICE_ORACLE_ABI_FILE", self.DEFAULT_PATHS["GAS_PRICE_ORACLE_ABI_FILE"], "Gas Oracle ABI filename")
        self.GAS_PRICE_ORACLE_ABI = self._resolve_path_in_dir(self.ABI_DIR, gas_oracle_file, "Gas Price Oracle ABI", required=False)


        # --- Resolve Config File Paths (within UTILS_DIR or other specified dir) ---
        self.TOKEN_ADDRESSES = self._resolve_path_in_dir(
            self.UTILS_DIR,
            self._get_env_str("TOKEN_ADDRESSES_FILE", self.DEFAULT_PATHS["TOKEN_ADDRESSES_FILE"], "Token Addresses filename"),
            "Token Addresses JSON", required=True
        )
        self.TOKEN_SYMBOLS = self._resolve_path_in_dir(
             self.UTILS_DIR,
             self._get_env_str("TOKEN_SYMBOLS_FILE", self.DEFAULT_PATHS["TOKEN_SYMBOLS_FILE"], "Token Symbols filename"),
             "Token Symbols JSON", required=True
         )
        # Optional ERC20 Signatures
        erc20_sig_file = self._get_env_str("ERC20_SIGNATURES_FILE", self.DEFAULT_PATHS["ERC20_SIGNATURES_FILE"], "ERC20 Signatures filename")
        self.ERC20_SIGNATURES = self._resolve_path_in_dir(self.UTILS_DIR, erc20_sig_file, "ERC20 Signatures JSON", required=False)

        # --- Load Contract Addresses ---
        for addr_key, default_addr in self.DEFAULT_ADDRESSES.items():
             # Address loading requires the validator from ConfigCore, passed during init if needed,
             # or use the internal basic validator. Full validation is in ConfigCore._validate_critical_settings.
             is_required = addr_key in ["AAVE_FLASHLOAN_ADDRESS"] # Example of a required address
             address_val = self._get_env_str(addr_key, default_addr, f"{addr_key} Contract Address", required=is_required)
             # Perform basic validation here, full checksumming in ConfigCore
             if address_val:
                 try:
                      validated_addr = self._validate_address_func(address_val, addr_key)
                      setattr(self, addr_key, validated_addr)
                 except ValueError as e:
                      # Log error but defer raising to ConfigCore validation stage
                      logger.error("Invalid address format for %s: %s. Fix environment variable.", addr_key, e)
                      setattr(self, addr_key, None) # Set to None if invalid format
             else:
                  setattr(self, addr_key, None) # Set to None if not provided and not required


        logger.debug("Configuration paths and addresses loaded.")


    def _resolve_path(self, env_var_name: str, default_rel_path: str, is_dir: bool = False, required: bool = False) -> Optional[Path]:
        """Resolves a path from env var or default, relative to base path."""
        path_str = self._get_env_str(env_var_name, default_rel_path, f"{env_var_name} Path", required=required)
        if path_str is None:
            return None # Not required and not set

        path = Path(path_str)
        if not path.is_absolute():
            path = (self._base_path / path).resolve()

        # Create directory if it's supposed to be one and doesn't exist
        if is_dir:
             path.mkdir(parents=True, exist_ok=True)
             logger.debug("Resolved/Ensured directory %s: %s", env_var_name, path)
        else:
             # Check file existence only if required
             if required and not path.exists():
                  logger.error("Required file not found for %s at resolved path: %s", env_var_name, path)
                  raise FileNotFoundError(f"Required file for {env_var_name} not found: {path}")
             elif not path.exists():
                  logger.debug("Resolved optional path %s: %s (File does not exist)", env_var_name, path)
             else:
                  logger.debug("Resolved path %s: %s", env_var_name, path)

        return path

    def _resolve_path_in_dir(self, dir_path: Path, filename: Optional[str], description: str, required: bool = False) -> Optional[Path]:
        """Resolves a file path within a specific directory."""
        if not filename:
             if required:
                  logger.error("Required filename for %s is missing.", description)
                  raise ValueError(f"Required filename for {description} missing.")
             return None # Optional file not specified

        file_path = (dir_path / filename).resolve()

        if required and not file_path.exists():
            logger.error("Required file '%s' not found in %s (Resolved: %s)", filename, dir_path, file_path)
            raise FileNotFoundError(f"Required {description} file '{filename}' not found in {dir_path}")
        elif not file_path.exists():
            logger.debug("Optional file '%s' for %s not found at %s.", filename, description, file_path)
            return None # Optional file doesn't exist
        else:
             logger.debug("Resolved path for %s: %s", description, file_path)
             return file_path
