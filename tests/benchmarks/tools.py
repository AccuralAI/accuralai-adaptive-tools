"""File operation tools for benchmark testing."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List


def get_benchmark_tools(test_data_dir: str) -> List[Dict[str, Any]]:
    """Get file operation tools for benchmark.
    
    Args:
        test_data_dir: Directory containing test data files
        
    Returns:
        List of tool definitions compatible with Google Gemini function calling
    """
    test_data_path = Path(test_data_dir)
    
    return [
        {
            "function_declaration": {
                "name": "read_file",
                "description": "Read the contents of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file to read, relative to test_data directory"
                        }
                    },
                    "required": ["file_path"]
                }
            }
        },
        {
            "function_declaration": {
                "name": "write_file",
                "description": "Write content to a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file to write, relative to test_data directory"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write to the file"
                        }
                    },
                    "required": ["file_path", "content"]
                }
            }
        },
        {
            "function_declaration": {
                "name": "list_files",
                "description": "List all files in the test_data directory",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        },
        {
            "function_declaration": {
                "name": "parse_csv",
                "description": "Parse a CSV file and return structured data",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to CSV file"
                        }
                    },
                    "required": ["file_path"]
                }
            }
        },
        {
            "function_declaration": {
                "name": "parse_json",
                "description": "Parse a JSON file and return structured data",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to JSON file"
                        }
                    },
                    "required": ["file_path"]
                }
            }
        },
    ]


class FileToolHandler:
    """Handler for file operation tools."""
    
    def __init__(self, test_data_dir: str):
        """Initialize handler.
        
        Args:
            test_data_dir: Directory containing test data files
        """
        self.test_data_dir = Path(test_data_dir)
        self.test_data_dir.mkdir(parents=True, exist_ok=True)
        self.call_count = {}  # Track tool calls for pattern detection
        
    async def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Handle a tool call.
        
        Args:
            tool_name: Name of the tool being called
            arguments: Tool arguments
            
        Returns:
            Tool result
        """
        # Track call for pattern detection
        self.call_count[tool_name] = self.call_count.get(tool_name, 0) + 1
        
        if tool_name == "read_file":
            file_path = self.test_data_dir / arguments["file_path"]
            if not file_path.exists():
                return {"error": f"File not found: {file_path}"}
            return {"content": file_path.read_text(), "path": str(file_path)}
            
        elif tool_name == "write_file":
            file_path = self.test_data_dir / arguments["file_path"]
            file_path.write_text(arguments["content"])
            return {"success": True, "path": str(file_path), "bytes_written": len(arguments["content"])}
            
        elif tool_name == "list_files":
            files = [f.name for f in self.test_data_dir.iterdir() if f.is_file()]
            return {"files": files, "count": len(files)}
            
        elif tool_name == "parse_csv":
            file_path = self.test_data_dir / arguments["file_path"]
            if not file_path.exists():
                return {"error": f"File not found: {file_path}"}
            content = file_path.read_text()
            lines = content.strip().split("\n")
            headers = lines[0].split(",")
            rows = []
            for line in lines[1:]:
                if line.strip():
                    values = line.split(",")
                    rows.append(dict(zip(headers, values)))
            return {"headers": headers, "rows": rows, "count": len(rows)}
            
        elif tool_name == "parse_json":
            file_path = self.test_data_dir / arguments["file_path"]
            if not file_path.exists():
                return {"error": f"File not found: {file_path}"}
            content = file_path.read_text()
            data = json.loads(content)
            return {"data": data}
            
        else:
            return {"error": f"Unknown tool: {tool_name}"}
    
    def get_call_stats(self) -> Dict[str, int]:
        """Get statistics about tool calls."""
        return dict(self.call_count)

