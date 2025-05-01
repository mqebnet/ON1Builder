# abiregistry.py
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
        self._initialized = False

    async def initialize(self, base_path: Optional[Path] = None) -> None:
        if self._initialized:
            return
        await self._load_all_abis(base_path)
        self._initialized = True

    async def _load_all_abis(self, base_path: Optional[Path] = None) -> None:
        if not base_path:
            base_path = Path(__file__).parent.parent
        abi_dir = base_path / "abi"
        abi_files = {
            "erc20": "erc20_abi.json",
            "uniswap": "uniswap_abi.json",
            "sushiswap": "sushiswap_abi.json",
            "aave_flashloan": "aave_flashloan_abi.json",
            "aave": "aave_pool_abi.json"
        }
        tasks = []
        for abi_type, filename in abi_files.items():
            tasks.append(self._load_and_validate_abi(abi_type, abi_dir / filename, {"erc20", "uniswap"}))
        await asyncio.gather(*tasks)

    async def _load_and_validate_abi(self, abi_type: str, abi_path: Path, critical_abis: set) -> None:
        try:
            abi = await self._load_abi_from_path(abi_path, abi_type)
            self.abis[abi_type] = abi
            self._extract_signatures(abi, abi_type)
        except Exception as e:
            logger.error(f"Error loading {abi_type}: {e}")
            if abi_type in critical_abis:
                raise

    async def _load_abi_from_path(self, abi_path: Path, abi_type: str) -> List[Dict[str, Any]]:
        if not abi_path.exists():
            raise FileNotFoundError(f"ABI not found: {abi_path}")
        async with aiofiles.open(abi_path, "r", encoding="utf-8") as f:
            content = await f.read()
        abi = json.loads(content)
        if not self._validate_abi(abi, abi_type):
            raise ValueError(f"Validation failed for {abi_type}")
        return abi

    def _validate_abi(self, abi: Any, abi_type: str) -> bool:
        if not isinstance(abi, list):
            return False
        found = {entry.get("name") for entry in abi if entry.get("type") == "function" and entry.get("name")}
        missing = self.REQUIRED_METHODS.get(abi_type, set()) - found
        return not missing

    def _extract_signatures(self, abi: List[Dict[str, Any]], abi_type: str) -> None:
        sigs: Dict[str, str] = {}
        selectors: Dict[str, str] = {}
        for entry in abi:
            if entry.get("type") == "function" and entry.get("name"):
                name = entry["name"]
                types = ",".join(inp.get("type", "") for inp in entry.get("inputs", []))
                signature = f"{name}({types})"
                selector = function_signature_to_4byte_selector(signature).hex()
                sigs[name] = signature
                selectors[selector] = name
        self.signatures[abi_type] = sigs
        self.method_selectors[abi_type] = selectors

    def get_abi(self, abi_type: str) -> Optional[List[Dict[str, Any]]]:
        return self.abis.get(abi_type)

    def get_method_selector(self, selector: str) -> Optional[str]:
        for sel in self.method_selectors.values():
            if selector in sel:
                return sel[selector]
        return None

    def get_function_signature(self, abi_type: str, method_name: str) -> Optional[str]:
        return self.signatures.get(abi_type, {}).get(method_name)

    async def update_abi(self, abi_type: str, new_abi: List[Dict[str, Any]]) -> None:
        if not self._validate_abi(new_abi, abi_type):
            raise ValueError(f"Validation failed for {abi_type}")
        self.abis[abi_type] = new_abi
        self._extract_signatures(new_abi, abi_type)

    async def validate_abi(self, abi_type: str) -> bool:
        abi = self.abis.get(abi_type)
        if not abi:
            return False
        return self._validate_abi(abi, abi_type)
