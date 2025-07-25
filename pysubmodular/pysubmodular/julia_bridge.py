"""
Julia bridge for PySubmodular.

This module handles the interface between Python and Julia, providing
automatic Julia setup, package installation, and error handling.
"""

import os
import sys
import warnings
from typing import Optional, Any, Dict, Callable
from pathlib import Path

import numpy as np

from .exceptions import JuliaError, PySubmodularError


class JuliaBridge:
    """
    Manages the Python-Julia interface for SubmodularMinimization.jl.
    
    This class handles Julia initialization, package loading, and provides
    a clean interface for calling Julia functions from Python.
    """
    
    _instance: Optional['JuliaBridge'] = None
    _julia = None
    _sm_module = None
    
    def __new__(cls) -> 'JuliaBridge':
        """Singleton pattern to ensure only one Julia instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        """Initialize Julia bridge if not already done."""
        if self._julia is None:
            self._initialize_julia()
    
    def _initialize_julia(self) -> None:
        """Initialize Julia and load SubmodularMinimization.jl."""
        try:
            import julia
            from julia import Main
            
            # Configure Julia
            self._julia = julia.Julia(compiled_modules=False)
            self._main = Main
            
            # Find the Julia package path
            package_path = self._find_package_path()
            
            # Activate and load the package
            self._main.eval(f'using Pkg; Pkg.activate("{package_path}")')
            self._main.eval('using SubmodularMinimization')
            
            # Import commonly used Julia functions
            # Access the module using eval
            self._sm_module = self._main.eval('SubmodularMinimization')
            
            # Test that everything works
            self._test_julia_connection()
            
        except ImportError as e:
            raise JuliaError(
                "PyJulia not installed. Please install with: pip install julia",
                e
            )
        except Exception as e:
            raise JuliaError(f"Failed to initialize Julia: {e}", e)
    
    def _find_package_path(self) -> str:
        """Find the path to SubmodularMinimization.jl package."""
        # Look for the package relative to this Python package
        current_dir = Path(__file__).parent.parent.parent
        
        # Check if we're in the development directory
        if (current_dir / "src" / "SubmodularMinimization.jl").exists():
            return str(current_dir)
        
        # Otherwise, assume it's installed as a Julia package
        # Let Julia handle the package location
        try:
            self._main.eval('using SubmodularMinimization')
            return ""  # Empty string means use current environment
        except:
            raise JuliaError(
                "SubmodularMinimization.jl not found. "
                "Please ensure it's installed or run from the development directory."
            )
    
    def _test_julia_connection(self) -> None:
        """Test that Julia connection is working properly."""
        try:
            # Simple test - create a small function and minimize it
            self._main.f_test = self._main.eval('ConcaveSubmodularFunction(3, 0.5)')
            result = self._main.eval('fujishige_wolfe_submodular_minimization(f_test)')
            
            # Check that we got a reasonable result tuple
            if not hasattr(result, '__len__') or len(result) != 4:
                raise JuliaError("Unexpected result format from Julia")
                
        except Exception as e:
            raise JuliaError(f"Julia connection test failed: {e}", e)
    
    def call_julia_function(self, func_name: str, *args, **kwargs) -> Any:
        """
        Call a Julia function with error handling.
        
        Args:
            func_name: Name of the Julia function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Result from Julia function call
            
        Raises:
            JuliaError: If Julia function call fails
        """
        try:
            # Get the Julia function
            func = self._main.eval(func_name)
            
            # Call it directly with arguments
            if kwargs:
                # Handle keyword arguments by combining with positional args
                # This is a simplified approach - for full compatibility we'd need
                # to handle Julia's keyword argument syntax
                return func(*args, **kwargs)
            else:
                return func(*args)
                
        except Exception as e:
            raise JuliaError(f"Julia function '{func_name}' failed: {e}", e)
    
    def _format_julia_value(self, value: Any) -> str:
        """Format a Python value for Julia eval."""
        if isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, bool):
            return "true" if value else "false"
        else:
            return str(value)
    
    def convert_python_array(self, arr: np.ndarray) -> Any:
        """Convert Python numpy array to Julia array."""
        try:
            # Julia uses column-major order, numpy uses row-major by default
            if arr.ndim > 1:
                arr = np.asfortranarray(arr)
            return self._main.eval("identity")(arr)
        except Exception as e:
            raise JuliaError(f"Failed to convert array to Julia: {e}", e)
    
    def convert_julia_array(self, julia_arr: Any) -> np.ndarray:
        """Convert Julia array to Python numpy array."""
        try:
            return np.array(julia_arr)
        except Exception as e:
            raise JuliaError(f"Failed to convert Julia array: {e}", e)
    
    def create_julia_bitvector(self, bool_array: np.ndarray) -> Any:
        """Create Julia BitVector from Python boolean array."""
        try:
            if bool_array.dtype != bool:
                bool_array = bool_array.astype(bool)
            return self._main.eval("BitVector")(bool_array)
        except Exception as e:
            raise JuliaError(f"Failed to create Julia BitVector: {e}", e)
    
    @property
    def julia(self) -> Any:
        """Access to raw Julia Main module."""
        return self._main
    
    @property
    def submodular_module(self) -> Any:
        """Access to SubmodularMinimization.jl module."""
        return self._sm_module
    
    def call_python_function(self, func_id: str, julia_bitvector) -> float:
        """Callback function for Julia to call Python submodular functions."""
        try:
            if not hasattr(self, '_python_functions'):
                raise JuliaError(f"No Python functions registered")
            
            if func_id not in self._python_functions:
                raise JuliaError(f"Python function {func_id} not found")
            
            func = self._python_functions[func_id]
            
            # Convert Julia BitVector to Python boolean array
            python_subset = np.array(julia_bitvector, dtype=bool)
            
            # Call the Python function
            result = func.evaluate(python_subset)
            return float(result)
            
        except Exception as e:
            raise JuliaError(f"Python function callback failed: {e}", e)
    
    def call_python_oracle(self, oracle_id: str, julia_c) -> Any:
        """Callback function for Julia to call Python linear oracles."""
        try:
            if not hasattr(self, '_python_oracles'):
                raise JuliaError(f"No Python oracles registered")
            
            if oracle_id not in self._python_oracles:
                raise JuliaError(f"Python oracle {oracle_id} not found")
            
            oracle = self._python_oracles[oracle_id]
            
            # Convert Julia array to Python numpy array
            python_c = np.array(julia_c, dtype=float)
            
            # Call the Python oracle
            result = oracle(python_c)
            return self.convert_python_array(result)
            
        except Exception as e:
            raise JuliaError(f"Python oracle callback failed: {e}", e)


# Global bridge instance
_bridge: Optional[JuliaBridge] = None


def get_julia_bridge() -> JuliaBridge:
    """Get the global Julia bridge instance."""
    global _bridge
    if _bridge is None:
        _bridge = JuliaBridge()
    return _bridge