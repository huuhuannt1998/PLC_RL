"""
PLC Logic Error Detection and Correction using Reinforcement Learning

This package provides tools for:
1. Parsing TIA Portal XML exports
2. Detecting logic errors in PLC code
3. Training RL agents to find and fix errors
4. Generating analysis reports

Usage:
    # Detect errors
    python error_detector.py
    
    # Train RL agent
    python rl_trainer.py
    
    # Generate analysis report
    python analyzer.py
"""

__version__ = "1.0.0"
__author__ = "PLC_RL Team"

from error_detector import (
    PLCXMLParser,
    ErrorDetector,
    ErrorCorrector,
    LogicError,
    ErrorType,
    ErrorSeverity,
    analyze_file
)

from rl_trainer import (
    FeatureExtractor,
    ErrorFeatures,
    ErrorDetectionNetwork,
    RLAgent,
    ErrorDetectionEnvironment,
    train_rl_agent
)

__all__ = [
    'PLCXMLParser',
    'ErrorDetector',
    'ErrorCorrector',
    'LogicError',
    'ErrorType',
    'ErrorSeverity',
    'analyze_file',
    'FeatureExtractor',
    'ErrorFeatures',
    'ErrorDetectionNetwork',
    'RLAgent',
    'ErrorDetectionEnvironment',
    'train_rl_agent'
]
