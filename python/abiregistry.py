import asyncio
import aiofiles
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from eth_utils import function_signature_to_4byte_selector
from loggingconfig import setup_logging
import logging

logger = setup_logging("AbiRegistry", level=logging.INFO)

class ABIRegistry:
    """
    Centralized ABI registry that loads, validates, and extracts function signatures.

    Attributes:
        REQUIRED_METHODS (dict): Mapping of ABI types to sets of required method names.
        abis (dict): Dictionary of loaded ABIs by type.
        signatures (dict): For each ABI type, a mapping from method names to full signature strings.
        method_selectors (dict): For each ABI type, a mapping from 4-byte hex selectors to method names.
    """
    REQUIRED_METHODS = {
        'erc20': {'transfer', 'approve', 'transferFrom', 'balanceOf'},
        'uniswap': {'swapExactTokensForTokens', 'swapTokensForExactTokens', 'addLiquidity', 'getAmountsOut'},
        'sushiswap': {'swapExactTokensForTokens', 'swapTokensForExactTokens', 'addLiquidity', 'getAmountsOut'},
        'aave_flashloan': {'fn_RequestFlashLoan', 'executeOperation', 'ADDRESSES_PROVIDER', 'POOL'},
        'aave': {'admin', 'implementation', 'upgradeTo', 'upgradeToAndCall'}
    }

    def __init__(self) -> None:
        self.abis: Dict[str, List[Dict[str, Any]]] = {}
        self.signatures: Dict[str, Dict[str, str]] = {}
        self.method_selectors: Dict[str, Dict[str, str]] = {}
        self._initialized: bool = False

    async def initialize(self, base_path: Optional[Path] = None) -> None:
        """
        Asynchronously load and validate all ABIs from the 'abi' directory.
        
        Args:
            base_path (Optional[Path]): Base path where the ABI folder is located.
                                         Defaults to the parent of the parent directory.
        """
        if self._initialized:
            logger.debug("ABIRegistry already initialized.")
            return

        await self._load_all_abis(base_path)
        self._initialized = True
        logger.debug("ABIRegistry initialization complete.")

    async def _load_all_abis(self, base_path: Optional[Path] = None) -> None:
        """
        Load all ABIs from the expected files in the 'abi' directory.
        """
        if not base_path:
            base_path = Path(__file__).parent.parent
        abi_dir = base_path / "abi"
        logger.debug(f"Loading ABIs from directory: {abi_dir}")

        # Define expected ABI files by type.
        abi_files = {
            "erc20": "erc20_abi.json",
            "uniswap": "uniswap_abi.json",
            "sushiswap": "sushiswap_abi.json",
            "aave_flashloan": "aave_flashloan_abi.json",
            "aave": "aave_pool_abi.json"
        }
        # Critical ABIs that must be loaded (if one fails, we want to stop).
        critical_abis = {"erc20", "uniswap"}

        tasks = []
        for abi_type, filename in abi_files.items():
            filepath = abi_dir / filename
            tasks.append(self._load_and_validate_abi(abi_type, filepath, critical_abis))
        await asyncio.gather(*tasks)

    async def _load_and_validate_abi(self, abi_type: str, abi_path: Path, critical_abis: set) -> None:
        """
        Load a single ABI file and validate its contents.
        """
        try:
            abi = await self._load_abi_from_path(abi_path, abi_type)
            self.abis[abi_type] = abi
            self._extract_signatures(abi, abi_type)
            logger.info(f"Loaded and validated {abi_type} ABI from {abi_path}")
        except Exception as e:
            logger.error(f"Error loading {abi_type} ABI from {abi_path}: {e}")
            if abi_type in critical_abis:
                raise
            else:
                logger.warning(f"Skipping non-critical ABI: {abi_type}")

    async def _load_abi_from_path(self, abi_path: Path, abi_type: str) -> List[Dict[str, Any]]:
        """
        Asynchronously load an ABI from a JSON file.
        
        Args:
            abi_path (Path): The path to the ABI file.
            abi_type (str): The ABI type.
        
        Returns:
            List[Dict[str, Any]]: The loaded ABI.
        
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If JSON decoding or validation fails.
        """
        if not abi_path.exists():
            msg = f"ABI file for {abi_type} not found at {abi_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        try:
            async with aiofiles.open(abi_path, "r", encoding="utf-8") as f:
                content = await f.read()
            abi = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {abi_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error reading ABI from {abi_path}: {e}")

        if not self._validate_abi(abi, abi_type):
            raise ValueError(f"ABI validation failed for {abi_type} at {abi_path}")
        return abi

    def _validate_abi(self, abi: List[Dict[str, Any]], abi_type: str) -> bool:
        """
        Validate that the ABI is a list and contains required methods.

        Args:
            abi (List[Dict[str, Any]]): The ABI to validate.
            abi_type (str): The type/category of the ABI.

        Returns:
            bool: True if valid, else False.
        """
        if not isinstance(abi, list):
            logger.error(f"ABI for {abi_type} is not a list.")
            return False
        
        found_methods = {entry.get("name") for entry in abi if entry.get("type") == "function" and entry.get("name")}
        required = self.REQUIRED_METHODS.get(abi_type, set())
        missing = required - found_methods
        if missing:
            logger.error(f"ABI for {abi_type} is missing required methods: {missing}")
            return False
        return True

    def _extract_signatures(self, abi: List[Dict[str, Any]], abi_type: str) -> None:
        """
        Extract full function signatures and their 4-byte selectors from the ABI.

        Args:
            abi (List[Dict[str, Any]]): The ABI.
            abi_type (str): The ABI type.
        """
        sigs = {}
        selectors = {}

        for entry in abi:
            if entry.get("type") == "function" and entry.get("name"):
                func_name = entry["name"]
                inputs = entry.get("inputs", [])
                types = ",".join(inp.get("type", "") for inp in inputs)
                signature = f"{func_name}({types})"
                selector = function_signature_to_4byte_selector(signature).hex()
                sigs[func_name] = signature
                selectors[selector] = func_name

        self.signatures[abi_type] = sigs
        self.method_selectors[abi_type] = selectors

    def get_abi(self, abi_type: str) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve the ABI for a given type.
        """
        return self.abis.get(abi_type)

    def get_method_selector(self, selector: str) -> Optional[str]:
        """
        Get the method name corresponding to a 4-byte selector across all ABIs.
        """
        for selectors in self.method_selectors.values():
            if selector in selectors:
                return selectors[selector]
        return None

    def get_function_signature(self, abi_type: str, method_name: str) -> Optional[str]:
        """
        Retrieve the full function signature for a given ABI type and method.
        """
        return self.signatures.get(abi_type, {}).get(method_name)

    async def update_abi(self, abi_type: str, new_abi: List[Dict[str, Any]]) -> None:
        """
        Update an ABI with new data dynamically.

        Args:
            abi_type (str): The type to update.
            new_abi (List[Dict[str, Any]]): The new ABI.

        Raises:
            ValueError: If the new ABI fails validation.
        """
        if not self._validate_abi(new_abi, abi_type):
            raise ValueError(f"Validation failed for {abi_type} ABI update.")
        self.abis[abi_type] = new_abi
        self._extract_signatures(new_abi, abi_type)
        logger.info(f"Updated {abi_type} ABI dynamically.")

    async def validate_abi(self, abi_type: str) -> bool:
        """
        Validate the ABI for a given type.

        Args:
            abi_type (str): The type of the ABI to validate.

        Returns:
            bool: True if the ABI is valid, else False.
        """
        abi = self.abis.get(abi_type)
        if not abi:
            logger.error(f"ABI for {abi_type} not found.")
            return False
        return self._validate_abi(abi, abi_type)
