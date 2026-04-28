"""
IO-Aware Operator Design Environment for Agents.

Provides a symbolic IO complexity calculator, computation graph DSL,
and design action space for agents to discover IO-efficient GPU operators.
"""

from .dsl import TensorSpec, Op, FusedOp, ComputationGraph, TilingSpec, OnlineStateSpec
from .calculator import IOCalculator, IOReport, HardwareSpec
from .actions import DesignActions
from .agent_loop import RuleBasedAgent
