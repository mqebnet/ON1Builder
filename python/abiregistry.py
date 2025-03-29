# -*- coding: utf-8 -*-
"""
ABIRegistry Module

A centralized registry for loading, validating, and mapping ABIs. It supports asynchronous
loading of ABIs from JSON files, validates that each ABI contains required methods, and
extracts function signatures and selectors for later use.
"""

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
    Centralized ABI registry with methods to load, validate, and extract function signatures.
    
    Attributes:
        REQUIRED_METHODS (dict): A mapping of ABI types to the set of required method names.
        abis (dict): Loaded and validated ABIs.
        signatures (dict): Mapping of ABI type to a dict of method names and their full signatures.
        method_selectors (dict): Mapping of ABI type to a dict of 4-byte selectors and method names.
    """

    REQUIRED_METHODS = {
        'erc20': {'transfer', 'approve', 'transferFrom', 'balanceOf'},
        'uniswap': {'swapExactTokensForTokens', 'swapTokensForExactTokens', 'addLiquidity', 'getAmountsOut'},
        'sushiswap': {'swapExactTokensForTokens', 'swapTokensForExactTokens', 'addLiquidity', 'getAmountsOut'},
        'aave_flashloan': {'fn_RequestFlashLoan', 'executeOperation', 'ADDRESSES_PROVIDER', 'POOL'},
        'aave': {'admin', 'implementation', 'upgradeToAndCall'}
    }

    def __init__(self) -> None:
        """
        Initialize the ABIRegistry with empty data structures.
        """
        self.abis: Dict[str, List[Dict[str, Any]]] = {}
        self.signatures: Dict[str, Dict[str, str]] = {}
        self.method_selectors: Dict[str, Dict[str, str]] = {}
        self._initialized: bool = False

    async def initialize(self, base_path: Optional[Path] = None) -> None:
        """
        Asynchronously load and validate all ABIs from the 'abi' directory.

        Args:
            base_path (Optional[Path]): The base path where the 'abi' directory is located.
                                         Defaults to the parent of this file's directory.
        """
        if self._initialized:
            logger.debug("ABIRegistry already initialized.")
            return
        await self._load_all_abis(base_path)
        self._initialized = True
        logger.debug("ABIRegistry initialization complete.")

    async def _load_all_abis(self, base_path: Optional[Path] = None) -> None:
        """
        Load and validate all ABIs from JSON files located in the 'abi' directory.

        Args:
            base_path (Optional[Path]): The base path where the 'abi' directory is located.
        """
        if not base_path:
            base_path = Path(__file__).parent.parent
        abi_dir = base_path / 'abi'
        logger.debug(f"ABI Directory: {abi_dir}")

        abi_files = {
            'erc20': 'erc20_abi.json',
            'uniswap': 'uniswap_abi.json',
            'sushiswap': 'sushiswap_abi.json',
            'aave_flashloan': 'aave_flashloan_abi.json',
            'aave': 'aave_pool_abi.json'
        }

        critical_abis = {'erc20', 'uniswap'}

        await asyncio.gather(
            *[self._load_and_validate_abi(abi_type, abi_dir / filename, critical_abis)
              for abi_type, filename in abi_files.items()]
        )

    async def _load_and_validate_abi(self, abi_type: str, abi_path: Path, critical_abis: set) -> None:
        """
        Load and validate a single ABI from a file.

        Args:
            abi_type (str): The type/category of the ABI.
            abi_path (Path): The file path of the ABI JSON.
            critical_abis (set): Set of ABI types that are considered critical.

        Raises:
            FileNotFoundError: If a critical ABI file is missing.
            ValueError: If a critical ABI fails validation.
        """
        try:
            abi = await self._load_abi_from_path(abi_path, abi_type)
            self.abis[abi_type] = abi
            self._extract_signatures(abi, abi_type)
            logger.info(f"Loaded and validated {abi_type} ABI from {abi_path}")
        except FileNotFoundError:
            logger.error(f"ABI file not found for {abi_type}: {abi_path}")
            if abi_type in critical_abis:
                raise
            logger.warning(f"Skipping non-critical ABI: {abi_type}")
        except (ValueError, json.JSONDecodeError) as e:
            logger.error(f"Error validating {abi_type} ABI: {e}")
            if abi_type in critical_abis:
                raise
            logger.warning(f"Skipping non-critical ABI: {abi_type}")
        except Exception as e:
            logger.error(f"Unexpected error loading {abi_type} ABI from {abi_path}: {e}", exc_info=True)
            if abi_type in critical_abis:
                raise
            logger.warning(f"Skipping non-critical ABI: {abi_type}")

    async def _load_abi_from_path(self, abi_path: Path, abi_type: str) -> List[Dict[str, Any]]:
        """
        Load and validate ABI content from the specified file path.

        Args:
            abi_path (Path): The file path of the ABI.
            abi_type (str): The type/category of the ABI.

        Returns:
            List[Dict[str, Any]]: The loaded ABI.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If ABI validation fails.
        """
        try:
            if not abi_path.exists():
                logger.error(f"ABI file not found: {abi_path}")
                raise FileNotFoundError(f"ABI file not found: {abi_path}")

            async with aiofiles.open(abi_path, 'r', encoding='utf-8') as f:
                abi_content = await f.read()
                abi = json.loads(abi_content)
                logger.debug(f"ABI content loaded from {abi_path}")

            if not self._validate_abi(abi, abi_type):
                raise ValueError(f"Validation failed for {abi_type} ABI from file {abi_path}")

            return abi
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for {abi_type} in file {abi_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading ABI {abi_type} from {abi_path}: {e}", exc_info=True)
            raise

    def _validate_abi(self, abi: List[Dict[str, Any]], abi_type: str) -> bool:
        """
        Validate the structure and required methods of an ABI.

        Args:
            abi (List[Dict[str, Any]]): The ABI to validate.
            abi_type (str): The type/category of the ABI.

        Returns:
            bool: True if the ABI is valid, else False.
        """
        if not isinstance(abi, list):
            logger.error(f"Invalid ABI format for {abi_type}: ABI must be a list.")
            return False

        found_methods = {
            item.get('name') for item in abi
            if item.get('type') == 'function' and 'name' in item
        }
        required = self.REQUIRED_METHODS.get(abi_type, set())
        if not required.issubset(found_methods):
            missing = required - found_methods
            logger.error(f"Missing required methods in {abi_type} ABI: {missing}. Required: {required}, Found: {found_methods}")
            return False

        return True

    def _extract_signatures(self, abi: List[Dict[str, Any]], abi_type: str) -> None:
        """
        Extract function signatures and 4-byte selectors from an ABI.

        Args:
            abi (List[Dict[str, Any]]): The ABI from which to extract data.
            abi_type (str): The type/category of the ABI.
        """
        signatures = {}
        selectors = {}

        for item in abi:
            if item.get('type') == 'function':
                name = item.get('name')
                if name:
                    inputs = ','.join(inp.get('type', '') for inp in item.get('inputs', []))
                    signature = f"{name}({inputs})"
                    selector = function_signature_to_4byte_selector(signature)
                    hex_selector = selector.hex()

                    signatures[name] = signature
                    selectors[hex_selector] = name

        self.signatures[abi_type] = signatures
        self.method_selectors[abi_type] = selectors

    def get_abi(self, abi_type: str) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve a validated ABI by type.

        Args:
            abi_type (str): The type of the ABI.

        Returns:
            Optional[List[Dict[str, Any]]]: The ABI if available, else None.
        """
        return self.abis.get(abi_type)

    def get_method_selector(self, selector: str) -> Optional[str]:
        """
        Retrieve a method name corresponding to a 4-byte selector.

        Args:
            selector (str): The 4-byte selector in hexadecimal format.

        Returns:
            Optional[str]: The method name if found, else None.
        """
        for selectors in self.method_selectors.values():
            if selector in selectors:
                return selectors[selector]
        return None

    def get_function_signature(self, abi_type: str, method_name: str) -> Optional[str]:
        """
        Retrieve the full function signature for a given ABI type and method name.

        Args:
            abi_type (str): The ABI type.
            method_name (str): The method name.

        Returns:
            Optional[str]: The function signature if available, else None.
        """
        return self.signatures.get(abi_type, {}).get(method_name)

    async def update_abi(self, abi_type: str, new_abi: List[Dict[str, Any]]) -> None:
        """
        Dynamically update an ABI without restarting the application.

        Args:
            abi_type (str): The ABI type to update.
            new_abi (List[Dict[str, Any]]): The new ABI content.

        Raises:
            ValueError: If the new ABI fails validation.
        """
        if not self._validate_abi(new_abi, abi_type):
            raise ValueError(f"Validation failed for {abi_type} ABI")
        self.abis[abi_type] = new_abi
        self._extract_signatures(new_abi, abi_type)
        logger.info(f"Updated {abi_type} ABI dynamically")
