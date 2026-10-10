"""Safe calculator and local Green Grid AI knowledge lookup tools."""

import ast
import math
import operator
from typing import Any


class ToolExecutionError(Exception):
    """Raised when tool execution fails."""


class SafeCalculator:
    """AST-based safe mathematical expression evaluator (no eval or arbitrary execution)."""

    ALLOWED_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    ALLOWED_FUNCTIONS = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sqrt": math.sqrt,
        "ceil": math.ceil,
        "floor": math.floor,
    }

    @classmethod
    def evaluate(cls, expression: str) -> int | float:
        """Safely evaluate an arithmetic expression string."""
        clean_expr = expression.strip().replace("$", "").replace(",", "")
        try:
            node = ast.parse(clean_expr, mode="eval")
        except SyntaxError as exc:
            raise ToolExecutionError(f"Invalid mathematical syntax: '{expression}'") from exc

        return cls._eval_node(node.body)

    @classmethod
    def _eval_node(cls, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ToolExecutionError(f"Non-numeric constant not allowed: {node.value}")

        elif isinstance(node, ast.BinOp):
            left = cls._eval_node(node.left)
            right = cls._eval_node(node.right)
            op_type = type(node.op)
            if op_type in cls.ALLOWED_OPERATORS:
                if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
                    raise ToolExecutionError("Division by zero is undefined")
                return cls.ALLOWED_OPERATORS[op_type](left, right)
            raise ToolExecutionError(f"Unsupported binary operator: {op_type.__name__}")

        elif isinstance(node, ast.UnaryOp):
            operand = cls._eval_node(node.operand)
            op_type = type(node.op)
            if op_type in cls.ALLOWED_OPERATORS:
                return cls.ALLOWED_OPERATORS[op_type](operand)
            raise ToolExecutionError(f"Unsupported unary operator: {op_type.__name__}")

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in cls.ALLOWED_FUNCTIONS:
                func = cls.ALLOWED_FUNCTIONS[node.func.id]
                args = [cls._eval_node(arg) for arg in node.args]
                return func(*args)
            raise ToolExecutionError(f"Unsupported function call: {getattr(node.func, 'id', 'unknown')}")

        raise ToolExecutionError(f"Unsupported expression node: {type(node).__name__}")


class GreenGridKnowledgeStore:
    """Curated knowledge repository for the Green Grid AI project."""

    DATA = {
        "summary": (
            "Green Grid AI is an intelligent commercial microgrid optimization system in Bengaluru deploying "
            "2.4 MW of rooftop solar paired with a 4.8 MWh lithium iron phosphate (LiFePO4) "
            "battery energy storage system (BESS). It performs automated real-time peak-shaving "
            "and time-of-use energy arbitrage."
        ),
        "solar_capacity": "2.4 MW commercial rooftop solar PV generating ~3,500,000 kWh annually.",
        "battery_storage": "4.8 MWh LiFePO4 battery storage system with 92% round-trip efficiency.",
        "tariffs": "Peak power tariff is $0.22/kWh; off-peak tariff is $0.09/kWh (spread: $0.13/kWh).",
        "arbitrage": (
            "Shifts 3,200 kWh/day from peak to off-peak hours, producing $416.00 in daily energy savings."
        ),
        "annual_savings": (
            "Annual energy arbitrage yields $151,840 (3,200 kWh * $0.13 * 365 days), plus an estimated "
            "$38,000 from a 15% peak demand charge reduction, bringing total annual savings to ~$189,840."
        ),
        "environmental_impact": "Prevents 2,450 metric tons of CO2 equivalents per year.",
        "status": "Phase 1 active across 4 commercial sites in Bengaluru with 99.4% operational uptime.",
    }

    @classmethod
    def lookup(cls, query: str) -> str:
        q = query.lower()
        results = []

        if any(w in q for w in ["summary", "about", "overview", "project", "what is", "green grid"]):
            results.append(cls.DATA["summary"])

        if any(w in q for w in ["solar", "generation", "pv", "capacity", "panel"]):
            results.append(cls.DATA["solar_capacity"])

        if any(w in q for w in ["battery", "storage", "bess", "lifepo4"]):
            results.append(cls.DATA["battery_storage"])

        if any(w in q for w in ["tariff", "peak", "off-peak", "rate", "cost"]):
            results.append(cls.DATA["tariffs"])

        if any(w in q for w in ["savings", "arbitrage", "calculate", "saving", "estimate", "roi", "dollar"]):
            results.append(cls.DATA["annual_savings"])

        if any(w in q for w in ["carbon", "co2", "emission", "greenhouse", "environment"]):
            results.append(cls.DATA["environmental_impact"])

        if any(w in q for w in ["status", "deploy", "site", "bengaluru", "uptime"]):
            results.append(cls.DATA["status"])

        if not results:
            return cls.DATA["summary"] + " " + cls.DATA["annual_savings"]

        return " ".join(results)
