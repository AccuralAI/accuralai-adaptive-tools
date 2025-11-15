"""PlanLang schema definitions and validation rules."""

from typing import Any, Dict, List


class PlanLangSchema:
    """PlanLang DSL schema definitions."""

    # Valid strategy types
    STRATEGY_TYPES = {
        "simple",      # No special handling
        "cached",      # Cache results with TTL
        "retry",       # Retry on failure with backoff
        "parallel",    # Execute in parallel with other steps
        "fallback",    # Try alternatives on failure
    }

    # Valid constraint types
    CONSTRAINT_TYPES = {
        "max_latency_ms",
        "max_cost_cents",
        "max_steps",
    }

    # Variable substitution pattern
    VARIABLE_PATTERN = r"\$\{([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\}"

    @staticmethod
    def get_base_schema() -> Dict[str, Any]:
        """Get JSON schema for PlanLang.

        Returns:
            JSON schema dictionary
        """
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["name", "version", "steps"],
            "properties": {
                "name": {
                    "type": "string",
                    "pattern": "^[a-z][a-z0-9_]*$",
                    "description": "Plan name (snake_case)"
                },
                "version": {
                    "type": "string",
                    "pattern": "^\\d+\\.\\d+\\.\\d+$",
                    "description": "Semantic version (X.Y.Z)"
                },
                "description": {
                    "type": "string",
                    "description": "Human-readable description"
                },
                "inputs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name"],
                        "properties": {
                            "name": {
                                "type": "string",
                                "pattern": "^[a-zA-Z_][a-zA-Z0-9_]*$"
                            },
                            "type": {
                                "type": "string",
                                "enum": ["string", "number", "boolean", "object", "array"]
                            },
                            "description": {"type": "string"},
                            "required": {"type": "boolean", "default": True},
                            "default": {"description": "Default value if not provided"}
                        }
                    }
                },
                "steps": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"$ref": "#/definitions/step"}
                },
                "constraints": {
                    "type": "object",
                    "properties": {
                        "max_latency_ms": {"type": "integer", "minimum": 0},
                        "max_cost_cents": {"type": "number", "minimum": 0},
                        "max_steps": {"type": "integer", "minimum": 1}
                    }
                },
                "eval_hooks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Validation hooks to run after execution"
                },
                "metadata": {
                    "type": "object",
                    "description": "Additional metadata"
                }
            },
            "definitions": {
                "step": {
                    "type": "object",
                    "required": ["id", "tool", "save_as"],
                    "properties": {
                        "id": {
                            "type": "string",
                            "pattern": "^[a-z][a-z0-9_]*$",
                            "description": "Step identifier"
                        },
                        "tool": {
                            "type": "string",
                            "description": "Tool name to execute"
                        },
                        "with_args": {
                            "type": "object",
                            "description": "Arguments to pass (can use ${var} substitution)"
                        },
                        "save_as": {
                            "type": "string",
                            "pattern": "^[a-zA-Z_][a-zA-Z0-9_]*$",
                            "description": "Variable name to store result"
                        },
                        "depends_on": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Step IDs this depends on"
                        },
                        "strategy": {
                            "oneOf": [
                                {"type": "string", "enum": list(PlanLangSchema.STRATEGY_TYPES)},
                                {"$ref": "#/definitions/strategy_config"}
                            ]
                        },
                        "conditional": {
                            "type": "string",
                            "description": "Condition to evaluate (${var} expressions)"
                        }
                    }
                },
                "strategy_config": {
                    "type": "object",
                    "required": ["type"],
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": list(PlanLangSchema.STRATEGY_TYPES)
                        },
                        "config": {
                            "type": "object",
                            "description": "Strategy-specific configuration"
                        }
                    }
                }
            }
        }

    @staticmethod
    def get_strategy_config_schema(strategy_type: str) -> Dict[str, Any]:
        """Get schema for specific strategy configuration.

        Args:
            strategy_type: Type of strategy

        Returns:
            Configuration schema
        """
        schemas = {
            "cached": {
                "type": "object",
                "properties": {
                    "ttl_seconds": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Cache TTL in seconds"
                    },
                    "key_template": {
                        "type": "string",
                        "description": "Cache key template (supports ${var})"
                    }
                }
            },
            "retry": {
                "type": "object",
                "properties": {
                    "max_attempts": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 3
                    },
                    "backoff": {
                        "type": "string",
                        "enum": ["linear", "exponential", "constant"],
                        "default": "exponential"
                    },
                    "initial_delay_ms": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 1000
                    },
                    "max_delay_ms": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 30000
                    }
                }
            },
            "parallel": {
                "type": "object",
                "properties": {
                    "max_concurrency": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Max parallel executions"
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Timeout for parallel execution"
                    }
                }
            },
            "fallback": {
                "type": "object",
                "properties": {
                    "alternatives": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "description": "List of alternative tool names"
                    }
                }
            }
        }

        return schemas.get(strategy_type, {"type": "object"})

    @staticmethod
    def get_example_plan() -> str:
        """Get example PlanLang YAML.

        Returns:
            Example YAML string
        """
        return """
name: "analyze_logs_optimized"
version: "1.0.0"
description: "Optimized log analysis with caching and parallel execution"

inputs:
  - name: "log_directory"
    type: "string"
    description: "Path to log directory"
    required: true
  - name: "pattern"
    type: "string"
    description: "Pattern to search for"
    default: "ERROR"

steps:
  - id: "list_logs"
    tool: "list.directory"
    with_args:
      path: "${inputs.log_directory}"
      pattern: "*.log"
    save_as: "log_files"
    strategy:
      type: "cached"
      config:
        ttl_seconds: 300

  - id: "search_pattern"
    tool: "grep.text"
    with_args:
      pattern: "${inputs.pattern}"
      files: "${log_files}"
    save_as: "matches"
    depends_on: ["list_logs"]
    strategy:
      type: "retry"
      config:
        max_attempts: 3
        backoff: "exponential"

  - id: "count_matches"
    tool: "count.lines"
    with_args:
      text: "${matches}"
    save_as: "count"
    depends_on: ["search_pattern"]

  - id: "format_output"
    tool: "format.json"
    with_args:
      data:
        log_directory: "${inputs.log_directory}"
        pattern: "${inputs.pattern}"
        matches_found: "${count}"
        files_searched: "${log_files}"
    save_as: "result"
    depends_on: ["count_matches"]

constraints:
  max_latency_ms: 5000
  max_cost_cents: 2

eval_hooks:
  - "validators.check_schema(result, ResultSchema)"
""".strip()
