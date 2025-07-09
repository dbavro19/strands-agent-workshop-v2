import streamlit as st
import json
import boto3
import os
from manage_tools import parse_bedrock_tool_calls, execute_tool_calls, get_tool_registry, format_tool_results_for_bedrock

# Set page configuration
st.set_page_config(
    page_title="LLM Function Calling Demo",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state variables if they don't exist
if "messages" not in st.session_state:
    st.session_state.messages = []
if "tools_enabled" not in st.session_state:
    st.session_state.tools_enabled = False
if "tool_calls" not in st.session_state:
    st.session_state.tool_calls = []
if "tool_results" not in st.session_state:
    st.session_state.tool_results = {}

# Function to initialize Bedrock client
@st.cache_resource
def get_bedrock_client():
    try:
        bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            region_name=("us-west-2")
        )
        return bedrock_runtime
    except Exception as e:
        st.error(f"Error initializing Bedrock client: {str(e)}")
        return None

# Function to call Bedrock model
def call_bedrock_model(messages, tools_enabled=False):
    client = get_bedrock_client()
    if not client:
        return "Error: Could not initialize Bedrock client"
    
    model_id = "anthropic.claude-3-5-haiku-20241022-v1:0"
    
    # Convert messages to the format expected by Bedrock
    formatted_messages = [
        {
            "role": message["role"],
            "content": [{"text": message["content"]}]
        } for message in messages
    ]
    
    # System prompt
    system_prompt = "You are a helpful assistant"
    
    # Basic request configuration
    body = {
        "modelId": model_id,
        "inferenceConfig": {
            "maxTokens": 1024,
            "temperature": 0.7
        },
        "system": [{"text": system_prompt}],
        "messages": formatted_messages
    }
    
    # Add tools configuration if enabled
    if tools_enabled:
        body["toolConfig"] = {
            "tools": [
                {
                    "toolSpec": {
                        "name": "random_number",
                        "description": "Generate a random integer between min and max values (inclusive)",
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {
                                    "min": {
                                        "type": "string",
                                        "description": "The minimum value (inclusive)"
                                    },
                                    "max": {
                                        "type": "string",
                                        "description": "The maximum value (inclusive)"
                                    }
                                },
                                "required": ["min", "max"]
                            }
                        }
                    }
                }
            ]
        }


    # Don't include toolConfig at all when tools are disabled
    
    try:
        # Make the API call
        response = client.converse(**body)
        
        # Process tool calls if tools are enabled
        if tools_enabled:
            tool_calls = parse_bedrock_tool_calls(response)
            st.session_state.tool_calls = tool_calls
    
            if tool_calls:
                # Execute tool calls
                tool_registry = get_tool_registry()
                tool_results = execute_tool_calls(tool_calls, tool_registry)
                st.session_state.tool_results = tool_results
        
                # Start a new conversation with just the original messages and the assistant's response
                new_conversation = formatted_messages.copy()
        
                # Add the assistant's response with tool calls
                assistant_message = {
                    "role": "assistant",
                    "content": response.get('output', {}).get('message', {}).get('content', [])
                }
                new_conversation.append(assistant_message)
        
                # Create a new user message with tool results in the correct format
                tool_results_content = []
        
                for tool_call in tool_calls:
                    tool_id = tool_call.get('id')
                    if tool_id in tool_results:
                        result = tool_results[tool_id]
                
                        # Create a toolResult block for each tool call
                        tool_result_block = {
                            "toolResult": {
                                "toolUseId": tool_id,
                                "content": [
                                    {
                                        "json": result if isinstance(result, dict) else {"result": result}
                                    }
                                ]
                            }
                        }
                        tool_results_content.append(tool_result_block)
        
                # Add a new user message with the tool results
                tool_results_message = {
                    "role": "user",
                    "content": tool_results_content
                }
        
                # Add the tool results message to the new conversation
                new_conversation.append(tool_results_message)
        
                # Create a new request with the updated conversation
                new_body = {
                    "modelId": model_id,
                    "inferenceConfig": {
                        "maxTokens": 1024,
                        "temperature": 0.7
                    },
                    "system": [{"text": system_prompt}],
                    "messages": new_conversation
                }
        
                # Add tool config to the new request
                if "toolConfig" in body:
                    new_body["toolConfig"] = body["toolConfig"]
        
                # Call Bedrock again with the tool results
                response = client.converse(**new_body)



        
        return response
    except Exception as e:
        st.error(f"Error calling Bedrock: {str(e)}")
        return None


# Function to extract text from Bedrock response
def extract_text_from_response(response):
    if not response:
        return "Error: No response received"
    
    try:
        message_content = response.get('output', {}).get('message', {}).get('content', [])
        text_parts = []
        
        for content_block in message_content:
            if 'text' in content_block:
                text_parts.append(content_block['text'])
        
        return ''.join(text_parts)
    except Exception as e:
        return f"Error extracting text: {str(e)}"

# Title and description
st.title("🤖 LLM Function Calling Demo")
st.markdown("""
This demo showcases the power of function calling in LLMs. Toggle the button below to enable or disable function calling
and see the difference in how the model handles tasks like generating random numbers.
""")

# Create two columns for the toggle and mode display
col1, col2 = st.columns([1, 3])

# Toggle for enabling/disabling tools
with col1:
    tools_enabled = st.toggle("Enable Function Calling", value=st.session_state.tools_enabled)
    if tools_enabled != st.session_state.tools_enabled:
        st.session_state.tools_enabled = tools_enabled
        st.rerun()

# Display current mode
with col2:
    if st.session_state.tools_enabled:
        st.success("Function calling is ENABLED. The model can use tools to perform tasks.")
    else:
        st.warning("Function calling is DISABLED. The model will try to simulate tasks without tools.")

# Chat interface
st.subheader("Chat with the Model")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input for new message
if prompt := st.chat_input("Ask the model to generate a random number or ask any question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Call Bedrock with or without tools based on toggle
            response = call_bedrock_model(
                [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                st.session_state.tools_enabled
            )
            
            # Display tool calls and results if tools are enabled and used
            if st.session_state.tools_enabled and st.session_state.tool_calls:
                with st.expander("🔧 Function Calling Process", expanded=True):
                    # Add a visual flow diagram of the process
                    st.markdown("### Function Calling Flow")
                    
                    # Create a 4-step flow diagram
                    flow_cols = st.columns(4)
                    
                    with flow_cols[0]:
                        st.markdown("#### 1. User Query")
                        st.info(f"💬 **User asks:** {prompt}")
                    
                    with flow_cols[1]:
                        st.markdown("#### 2. Model Identifies Tool")
                        tool_names = ", ".join([f"`{call['name']}`" for call in st.session_state.tool_calls])
                        st.success(f"🔍 **Model decides to use:** {tool_names}")
                    
                    with flow_cols[2]:
                        st.markdown("#### 3. Tool Execution")
                        for tool_call in st.session_state.tool_calls:
                            tool_id = tool_call.get('id')
                            if tool_id in st.session_state.tool_results:
                                result = st.session_state.tool_results[tool_id]
                                if isinstance(result, (int, float)):
                                    st.warning(f"⚙️ **Tool returns:** {result}")
                                else:
                                    st.warning(f"⚙️ **Tool executed successfully**")
                    
                    with flow_cols[3]:
                        st.markdown("#### 4. Final Response")
                        st.info("💡 **Model incorporates result**")
                    
                    # Progress bar to visualize the flow
                    st.progress(100)
                    
                    # Detailed tool calls and results
                    st.markdown("### Detailed Tool Information")
                    for i, tool_call in enumerate(st.session_state.tool_calls):
                        st.markdown(f"**Tool Call #{i+1}**")
                        
                        # Create columns for tool details
                        tc_col1, tc_col2 = st.columns(2)
                        
                        with tc_col1:
                            st.markdown("### Input")
                            st.markdown(f"**Tool Name:** `{tool_call['name']}`")
                            st.markdown("**Parameters:**")
                            st.json(tool_call['parameters'])
                        
                        with tc_col2:
                            # Display tool results
                            tool_id = tool_call.get('id')
                            if tool_id in st.session_state.tool_results:
                                st.markdown("### Output")
                                result = st.session_state.tool_results[tool_id]
                                
                                # Format the result for display
                                if isinstance(result, (int, float)):
                                    # For simple numeric results
                                    st.markdown(f"**Result:** {result}")
                                elif isinstance(result, dict) and "error" in result:
                                    # For error results
                                    st.error(f"Error: {result['error']}")
                                else:
                                    # For other results, try to display as JSON
                                    try:
                                        st.json(result)
                                    except:
                                        # Fallback to displaying as text
                                        st.markdown(f"**Result:** {str(result)}")
                        
                        st.divider()
            
            # Extract and display the final text response
            assistant_response = extract_text_from_response(response)
            st.markdown(assistant_response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            
            # Clear tool calls and results for next interaction
            st.session_state.tool_calls = []
            st.session_state.tool_results = {}


    
    # Add a reset button
    if st.button("Reset Chat"):
        st.session_state.messages = []
        st.session_state.tool_calls = []
        st.session_state.tool_results = {}
        st.rerun()
