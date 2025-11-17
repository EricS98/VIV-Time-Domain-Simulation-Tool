# common/terminal_formatting.py
"""
Terminal Formatting Utilities
==============================

Handles cross-platform terminal output with emoji support detection.
"""

import sys
import os
from typing import Dict

class TerminalFormatter:
    """
    Centralized terminal formatting with automatic emoji fallback.
    
    Detects terminal capabilities and provides appropriate symbols.
    """
    
    # Emoji mappings with ASCII fallbacks
    SYMBOLS = {
        'success': ('✅', '[OK]'),
        'error': ('❌', '[ERROR]'),
        'warning': ('⚠️', '[WARNING]'),
        'info': ('ℹ️', '[INFO]'),
        'target': ('🎯', '[*]'),
        'folder': ('📁', '[DIR]'),
        'file': ('🗂️', '[FILE]'),
        'arrow_right': ('→', '->'),
        'check': ('✓', 'v'),
        'cross': ('✗', 'x'),
    }
    
    def __init__(self, use_emoji: bool = None):
        """
        Initialize terminal formatter.
        
        Parameters:
        -----------
        use_emoji : bool, optional
            Force emoji on/off. If None, auto-detect terminal capability.
        """
        if use_emoji is None:
            self.use_emoji = self._detect_emoji_support()
        else:
            self.use_emoji = use_emoji
    
    @staticmethod
    def _detect_emoji_support() -> bool:
        """
        Detect if terminal supports emoji display.
        
        Detection strategy:
        1. Check environment variable TERM_EMOJI (explicit control)
        2. Check if running in Windows CMD (no emoji support)
        3. Check for common terminals with emoji support
        4. Default to True for modern terminals
        
        Returns:
        --------
        bool
            True if emoji should be used
        """
        # Explicit override via environment variable
        env_emoji = os.environ.get('TERM_EMOJI', '').lower()
        if env_emoji in ('0', 'false', 'no', 'off'):
            return False
        if env_emoji in ('1', 'true', 'yes', 'on'):
            return True
        
        # Check platform
        if sys.platform == 'win32':
            # Windows: Check if running in modern terminal
            # Windows Terminal, VSCode, PyCharm support emoji
            term_program = os.environ.get('TERM_PROGRAM', '').lower()
            wt_session = os.environ.get('WT_SESSION')  # Windows Terminal
            
            if wt_session or term_program in ('vscode', 'pycharm'):
                return True
            
            # Traditional CMD/PowerShell: no emoji
            return False
        
        # Unix-like systems: Check TERM variable
        term = os.environ.get('TERM', '').lower()
        
        # Terminals known to support emoji
        emoji_terms = ['xterm-256color', 'screen-256color', 'tmux-256color']
        if any(t in term for t in emoji_terms):
            return True
        
        # Check if in CI environment (usually no emoji)
        ci_env = any(os.environ.get(var) for var in ['CI', 'GITHUB_ACTIONS', 'GITLAB_CI', 'JENKINS_HOME'])
        if ci_env:
            return False
        
        # Default: assume modern terminal
        return True
    
    def get_symbol(self, name: str) -> str:
        """
        Get symbol (emoji or ASCII fallback).
        
        Parameters:
        -----------
        name : str
            Symbol name from SYMBOLS dict
            
        Returns:
        --------
        str
            Emoji or ASCII symbol
        """
        if name not in self.SYMBOLS:
            return name
        
        emoji, ascii_fallback = self.SYMBOLS[name]
        return emoji if self.use_emoji else ascii_fallback
    
    def format_message(self, message_type: str, message: str) -> str:
        """
        Format message with appropriate symbol.
        
        Parameters:
        -----------
        message_type : str
            Type: 'success', 'error', 'warning', 'info'
        message : str
            Message text
            
        Returns:
        --------
        str
            Formatted message
        """
        symbol = self.get_symbol(message_type)
        return f"{symbol} {message}"
    
    # Convenience methods
    def success(self, msg: str) -> str:
        return self.format_message('success', msg)
    
    def error(self, msg: str) -> str:
        return self.format_message('error', msg)
    
    def warning(self, msg: str) -> str:
        return self.format_message('warning', msg)
    
    def info(self, msg: str) -> str:
        return self.format_message('info', msg)


# Global instance for easy access
_formatter = None

def get_formatter() -> TerminalFormatter:
    """Get global terminal formatter instance."""
    global _formatter
    if _formatter is None:
        _formatter = TerminalFormatter()
    return _formatter

def set_emoji_enabled(enabled: bool):
    """Globally enable/disable emoji."""
    global _formatter
    _formatter = TerminalFormatter(use_emoji=enabled)

# Convenience functions for common use
def success(msg: str) -> str:
    return get_formatter().success(msg)

def error(msg: str) -> str:
    return get_formatter().error(msg)

def warning(msg: str) -> str:
    return get_formatter().warning(msg)

def info(msg: str) -> str:
    return get_formatter().info(msg)