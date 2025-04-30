import asyncio
import aiofiles
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from eth_utils import function_signature_to_4byte_selector
from loggingconfig import setup_logging
import logging

logger = setup_logging("AbiRegistry", level=logging.DEBUG)

class ABIRegistry:
    """
    A centralized registry for ABI data that loads, validates, and extracts
    function signatures and selectors.
    """
    # Define required methods for validation (adjust as needed)
    REQUIRED_METHODS = {
        'erc20': {'transfer', 'approve', 'transferFrom', 'balanceOf', 'totalSupply', 'decimals'},
        'uniswap': {'swapExactTokensForTokens', 'swapTokensForExactTokens', 'addLiquidity', 'removeLiquidity', 'getAmountsOut'},
        'sushiswap': {'swapExactTokensForTokens', 'swapTokensForExactTokens', 'addLiquidity', 'removeLiquidity', 'getAmountsOut'},
        'aave_flashloan': {'flashLoan', 'executeOperation', 'ADDRESSES_PROVIDER', 'POOL'}, # Corrected method names potentially
        'aave': {'supply', 'withdraw', 'borrow', 'repay', 'ADDRESSES_PROVIDER', 'POOL'} # Example methods
    }

    def __init__(self) -> None:
        self.abis: Dict[str, List[Dict[str, Any]]] = {}
        self.signatures: Dict[str, Dict[str, str]] = {} # method_name -> signature string
        self.method_selectors: Dict[str, Dict[str, str]] = {} # selector (0x...) -> method_name
        self._initialized = False
        self._base_path = Path(__file__).parent.parent # Default base path

    async def initialize(self, base_path: Optional[Path] = None) -> None:
        """
        Asynchronously load and validate all ABIs from the 'abi' directory.
        """
        if self._initialized:
            # A2: Parameterized logging
            logger.debug("ABIRegistry already initialized.")
            return

        if base_path:
            self._base_path = base_path
        elif not self._base_path:
             raise ValueError("Base path for ABIs not set.")

        await self._load_all_abis()
        self._initialized = True
        # A2: Parameterized logging
        logger.info("ABIRegistry initialization complete. Loaded ABIs for: %s", list(self.abis.keys()))

    async def _load_all_abis(self) -> None:
        """Loads all recognized ABI files from the standard ABI directory."""
        abi_dir = self._base_path / "abi"
        # A2: Parameterized logging
        logger.debug("Loading ABIs from: %s", abi_dir)

        # Standard ABI file names (adjust if structure differs)
        abi_files = {
            "erc20": "erc20_abi.json",
            "uniswap": "uniswap_abi.json", # Assuming Uniswap V2 Router
            "sushiswap": "sushiswap_abi.json", # Assuming SushiSwap Router
            "aave_flashloan": "aave_flashloan_abi.json", # Specify correct Aave Flashloan provider/receiver ABI
            "aave": "aave_pool_abi.json" # Assuming Aave V3 Pool ABI
            # Add other ABIs as needed
        }
        # Define which ABIs are critical for the bot to function
        critical_abis = {"erc20", "uniswap", "aave"} # Example

        tasks = []
        for abi_type, filename in abi_files.items():
            abi_path = abi_dir / filename
            tasks.append(self._load_and_validate_abi(abi_type, abi_path, abi_type in critical_abis))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result, (abi_type, _) in zip(results, abi_files.items()):
             if isinstance(result, Exception):
                 # A2: Parameterized logging
                 logger.error("Failed to load ABI '%s': %s", abi_type, result)
                 # If critical, re-raise the exception to halt initialization
                 if abi_type in critical_abis:
                     raise result
             else:
                 # A2: Parameterized logging
                 logger.debug("Successfully processed ABI '%s'.", abi_type)


    async def _load_and_validate_abi(self, abi_type: str, abi_path: Path, is_critical: bool) -> None:
        """Loads, validates, and processes a single ABI file."""
        # A2: Parameterized logging
        logger.debug("Attempting to load ABI '%s' from %s", abi_type, abi_path)
        try:
            abi = await self._load_abi_from_path(abi_path, abi_type)
            if self._validate_abi_structure(abi, abi_type) and self._validate_required_methods(abi, abi_type):
                self.abis[abi_type] = abi
                self._extract_signatures_and_selectors(abi, abi_type)
                # A2: Parameterized logging
                logger.info("Loaded and validated ABI '%s' (%d entries) from %s", abi_type, len(abi), abi_path)
            else:
                # Validation failed, raise error or log warning
                message = f"Validation failed for ABI '{abi_type}' at {abi_path}"
                if is_critical:
                    logger.critical(message)
                    raise ValueError(message)
                else:
                    logger.warning("%s. Skipping this non-critical ABI.", message)

        except FileNotFoundError:
            message = f"ABI file for '{abi_type}' not found at {abi_path}"
            if is_critical:
                logger.critical(message)
                raise
            else:
                 logger.warning("%s. Skipping this non-critical ABI.", message)
        except ValueError as e: # Catch JSON errors or validation errors
            message = f"Error loading/validating ABI '{abi_type}' from {abi_path}: {e}"
            if is_critical:
                logger.critical(message)
                raise
            else:
                logger.warning("%s. Skipping this non-critical ABI.", message)
        except Exception as e:
            # Catch any other unexpected errors during loading
            message = f"Unexpected error loading ABI '{abi_type}' from {abi_path}: {e}"
            if is_critical:
                 logger.critical(message, exc_info=True)
                 raise
            else:
                 logger.warning("%s. Skipping this non-critical ABI.", message, exc_info=True)


    async def _load_abi_from_path(self, abi_path: Path, abi_type: str) -> List[Dict[str, Any]]:
        """Loads ABI data from a JSON file."""
        if not await asyncio.to_thread(abi_path.exists):
            raise FileNotFoundError(f"ABI file not found at {abi_path}")
        try:
            async with aiofiles.open(abi_path, "r", encoding="utf-8") as f:
                content = await f.read()
            abi_data = json.loads(content)
            # Sometimes ABIs are nested (e.g., Truffle artifacts)
            if isinstance(abi_data, dict) and "abi" in abi_data:
                return abi_data["abi"]
            elif isinstance(abi_data, list):
                return abi_data
            else:
                 raise ValueError("Unexpected ABI format: expected a list or a dict with 'abi' key.")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {abi_path}: {e}")
        except Exception as e:
            # Catch potential OS errors during file read
            raise RuntimeError(f"Error reading ABI file {abi_path}: {e}")


    def _validate_abi_structure(self, abi: List[Dict[str, Any]], abi_type: str) -> bool:
        """Validates the basic structure of the ABI list."""
        if not isinstance(abi, list):
            # A2: Parameterized logging
            logger.error("ABI validation failed for '%s': Input is not a list.", abi_type)
            return False
        for i, entry in enumerate(abi):
            if not isinstance(entry, dict):
                # A2: Parameterized logging
                logger.error("ABI validation failed for '%s': Entry %d is not a dictionary.", abi_type, i)
                return False
            if entry.get("type") not in ["function", "constructor", "event", "fallback", "receive", "error"]:
                 # A2: Parameterized logging
                 logger.warning("ABI validation for '%s': Entry %d has unusual type '%s'.", abi_type, i, entry.get("type"))
            # Add more structural checks if needed (e.g., presence of 'name' for functions)
        return True

    def _validate_required_methods(self, abi: List[Dict[str, Any]], abi_type: str) -> bool:
        """Validates if required methods are present in the ABI."""
        required = self.REQUIRED_METHODS.get(abi_type)
        if not required:
            return True # No specific methods required for this type

        found_methods = {entry.get("name") for entry in abi if entry.get("type") == "function" and entry.get("name")}
        missing = required - found_methods

        if missing:
            # A2: Parameterized logging
            logger.error("ABI validation failed for '%s': Missing required methods: %s", abi_type, sorted(list(missing)))
            return False
        return True

    def _extract_signatures_and_selectors(self, abi: List[Dict[str, Any]], abi_type: str) -> None:
        """Extracts function signatures and 4-byte selectors from the ABI."""
        func_signatures: Dict[str, str] = {}
        func_selectors: Dict[str, str] = {}
        for entry in abi:
            # Process only functions with names
            if entry.get("type") == "function" and entry.get("name"):
                func_name = entry["name"]
                inputs = entry.get("inputs", [])
                # Construct the canonical signature string (e.g., "transfer(address,uint256)")
                input_types = ",".join(inp.get("type", "") for inp in inputs)
                signature = f"{func_name}({input_types})"

                try:
                    # Calculate the 4-byte selector
                    selector = function_signature_to_4byte_selector(signature).hex() # hex() includes '0x'
                    func_signatures[func_name] = signature
                    func_selectors[selector] = func_name # Map selector back to name
                except Exception as e:
                     # A2: Parameterized logging
                     logger.warning("Could not generate selector for signature '%s' in ABI '%s': %s", signature, abi_type, e)

        self.signatures[abi_type] = func_signatures
        self.method_selectors[abi_type] = func_selectors
        # A2: Parameterized logging
        logger.debug("Extracted %d signatures and %d selectors for ABI '%s'.", len(func_signatures), len(func_selectors), abi_type)


    def get_abi(self, abi_type: str) -> Optional[List[Dict[str, Any]]]:
        """Retrieves the ABI list for a given type."""
        if not self._initialized:
            logger.warning("Attempted to get ABI before ABIRegistry was initialized.")
            return None
        abi = self.abis.get(abi_type)
        if abi is None:
            # A2: Parameterized logging
            logger.debug("ABI type '%s' not found in registry.", abi_type)
        return abi

    def get_method_name_from_selector(self, selector: str) -> Optional[str]:
        """Finds the method name corresponding to a 4-byte selector across all loaded ABIs."""
        if not self._initialized:
            logger.warning("Attempted to get method name before ABIRegistry was initialized.")
            return None
        selector_lower = selector.lower()
        for type_selectors in self.method_selectors.values():
            if selector_lower in type_selectors:
                return type_selectors[selector_lower]
        # A2: Parameterized logging
        logger.debug("Selector '%s' not found in any loaded ABI.", selector)
        return None

    def get_function_signature(self, abi_type: str, method_name: str) -> Optional[str]:
        """Retrieves the full function signature string for a method in a specific ABI."""
        if not self._initialized:
            logger.warning("Attempted to get signature before ABIRegistry was initialized.")
            return None
        sig = self.signatures.get(abi_type, {}).get(method_name)
        if sig is None:
            # A2: Parameterized logging
            logger.debug("Method '%s' not found in signatures for ABI type '%s'.", method_name, abi_type)
        return sig

    async def update_abi(self, abi_type: str, new_abi_data: List[Dict[str, Any]]) -> bool:
        """Dynamically updates the ABI for a given type after validation."""
        # A2: Parameterized logging
        logger.info("Attempting to dynamically update ABI for '%s'.", abi_type)
        if self._validate_abi_structure(new_abi_data, abi_type) and self._validate_required_methods(new_abi_data, abi_type):
            self.abis[abi_type] = new_abi_data
            self._extract_signatures_and_selectors(new_abi_data, abi_type)
            # A2: Parameterized logging
            logger.info("Successfully updated ABI for '%s'.", abi_type)
            return True
        else:
            # A2: Parameterized logging
            logger.error("Dynamic update failed for ABI '%s' due to validation errors.", abi_type)
            return False

    async def validate_abi(self, abi_type: str) -> bool:
        """Checks if the currently loaded ABI for a type is valid."""
        if not self._initialized:
            logger.warning("Attempted to validate ABI before ABIRegistry was initialized.")
            return False
        abi = self.abis.get(abi_type)
        if not abi:
            # A2: Parameterized logging
            logger.error("Cannot validate ABI: Type '%s' not found.", abi_type)
            return False
        return self._validate_abi_structure(abi, abi_type) and self._validate_required_methods(abi, abi_type)