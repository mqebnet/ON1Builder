"""
0xBuilder package initialization.
"""
# ./0xBuilder/utils/Python/__init__.py

from strategyexecutionerror import StrategyExecutionError

__all__: list[str] = ['StrategyExecutionError'] 

def initialize_package() -> None:
    """
    Initialize the 0xBuilder package.
    """
    try:
        # Add any package initialization logic here
        pass
    except Exception as e:
        raise RuntimeError(f"Failed to initialize 0xBuilder package: {e}")

def cleanup_package() -> None:
    """
    Clean up resources used by the 0xBuilder package.
    """
    try:
        # Add any package cleanup logic here
        pass
    except Exception as e:
        raise RuntimeError(f"Failed to clean up 0xBuilder package: {e}")
