import json
from random_number_generator import random_number



tool_registry = {
    'random_number': random_number
}



def get_tool_registry():
    return tool_registry


def parse_bedrock_tool_calls(response):
    """
    Parse tool calls from a Bedrock converse API response.
    
    Args:
        response (dict): The complete response from Bedrock's converse API
        
    Returns:
        list: List of dictionaries containing tool calls with:
            - name: The tool name
            - parameters: Dictionary of parameter name/value pairs
    """
    tool_calls = []
    
    try:
        # Extract the message content from the response
        message_content = response.get('output', {}).get('message', {}).get('content', [])
        
        # Look for toolUse blocks in the content
        for content_block in message_content:
            if 'toolUse' in content_block:
                tool_use = content_block['toolUse']
                
                tool_name = tool_use.get('name')
                tool_input = tool_use.get('input', {})
                tool_use_id = tool_use.get('toolUseId')
                
                if tool_name:
                    tool_calls.append({
                        'id': tool_use_id,
                        'name': tool_name,
                        'parameters': tool_input
                    })
    
    except Exception as e:
        print(f"Error parsing tool calls: {str(e)}")
    
    return tool_calls


def execute_tool_calls(tool_calls, tool_registry):
    """
    Execute a list of tool calls and format the results.
    
    Args:
        tool_calls (list): List of tool call dictionaries from parse_bedrock_tool_calls
        tool_registry (dict): Dictionary mapping tool names to functions
        
    Returns:
        dict: Dictionary containing tool results with tool IDs as keys
    """
    tool_results = {}
    
    for tool_call in tool_calls:
        tool_id = tool_call.get('id')
        tool_name = tool_call.get('name')
        parameters = tool_call.get('parameters', {})
        
        if not tool_name or tool_name not in tool_registry:
            tool_results[tool_id] = {
                'error': f"Tool '{tool_name}' not found in registry"
            }
            continue
        
        try:
            # Execute the tool
            result = tool_registry[tool_name](**parameters)
            tool_results[tool_id] = result
        except Exception as e:
            tool_results[tool_id] = {
                'error': f"Error executing tool '{tool_name}': {str(e)}"
            }
    
    return tool_results

def format_tool_results_for_bedrock(tool_results):
    """
    Format tool results for sending back to Bedrock's converse API.
    
    Args:
        tool_results (dict): Dictionary of tool results from execute_tool_calls
        
    Returns:
        list: List of toolResult objects for Bedrock
    """
    formatted_results = []
    
    for tool_id, result in tool_results.items():
        # Convert dictionary result to a list of text blocks
        content_blocks = []
        if isinstance(result, dict):
            # Convert the dictionary to a single text block
            content_blocks.append({'text': json.dumps(result)})
        elif isinstance(result, list):
            # Already a list, keep as is
            content_blocks = result
        else:
            # Convert to a text block
            content_blocks.append({'text': str(result)})
        
        formatted_results.append({
            'toolUseId': tool_id,
            'content': content_blocks  # This must be a list
        })
    
    return formatted_results
