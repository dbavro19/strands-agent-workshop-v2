from mcp.server import FastMCP
import yfinance as yf

mcp = FastMCP("Yahoo Finance Server")

@mcp.tool(description="Get stock information from Yahoo Finance")
def get_stock_info(symbol: str) -> dict:
    """Get current stock price and basic info for a given symbol."""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        
        return {
            "symbol": symbol.upper(),
            "name": info.get("longName", "N/A"),
            "current_price": info.get("currentPrice", "N/A"),
            "previous_close": info.get("previousClose", "N/A"),
            "market_cap": info.get("marketCap", "N/A"),
            "volume": info.get("volume", "N/A")
        }
    except Exception as e:
        return {"error": f"Failed to fetch data for {symbol}: {str(e)}"}

mcp.run(transport="stdio")