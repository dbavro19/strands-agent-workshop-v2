from mcp.server import FastMCP
import random

mcp = FastMCP("RNG Server")

@mcp.tool(description="Generates a random Number using a minimum and maximum range")
def random_num(min: int = 1, max: int = 10) -> int:
    """Generate a random number between min and max (inclusive)."""

    return random.randint(min, max)

@mcp.tool(description="Gets the square root of a number")
def sqrt(num: int) -> float:
    """Get the square root of a number."""

    return num ** 0.5


mcp.run(transport="stdio")