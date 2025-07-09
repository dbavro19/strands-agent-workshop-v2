from strands import Agent
from strands.models import BedrockModel
from strands_tools import python_repl
#import python_repl_windows # Windows specific python repl 
from stock_tool import get_daily_historical_stock_data


bedrock_model = BedrockModel(
    model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",  # Specify the model ID
    temperature=0.7,  # Control randomness (0.0 to 1.0)
    top_p=0.9,  # Control diversity
    max_tokens=2000,  # Maximum response length
    region_name="us-west-2"  # AWS region where you have model access
)


#Agent WITH tje use_aws tool

#running the tool manually:


agent = Agent(
  model = bedrock_model,
  tools=[python_repl, get_daily_historical_stock_data], #use python_repl_windows if on windows
  system_prompt="""
You will be provided with a company
Retrieve the stock price history of that stock
Then create code to interpret the provided data
Then use the python_repl tool to generate and execute code to visualize the pricing history as a chart- the code generated must generate an image file to be saved locally
"""
)


#run the stock price history tool manually to show results
#tool_result = agent.tool.get_daily_historical_stock_data(symbol="AMZN")

#print(tool_result)


results =agent("Amazon")


