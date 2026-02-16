import streamlit as st
import requests
import openai
import json

# Set up the page
st.set_page_config(page_title="Lab 5: Weather Bot", page_icon=":partly_sunny:")
st.title("What to Wear Bot :partly_sunny:")

# Get API keys from secrets
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    openweathermap_api_key = st.secrets["OPENWEATHERMAP_API_KEY"]
except KeyError:
    st.error("Please set your API keys in .streamlit/secrets.toml")
    st.stop()

client = openai.OpenAI(api_key=openai_api_key)

if "weather_api_status" not in st.session_state:
    st.session_state.weather_api_status = None

def get_current_weather(location):
    """
    Get the current weather for a given location using OpenWeatherMap API.
    """
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={openweathermap_api_key}&units=imperial"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        weather_info = {
            "location": data["name"],
            "temperature": data["main"]["temp"],
            "description": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"]
        }
        st.session_state.weather_api_status = "Success"
        return json.dumps(weather_info)
    except requests.exceptions.RequestException as e:
        st.session_state.weather_api_status = f"Error: {e}"
        return json.dumps({"error": str(e)})
    except KeyError:
         st.session_state.weather_api_status = "Error: Invalid response format"
         return json.dumps({"error": "Location not found or invalid response format."})

# Debug Sidebar
with st.sidebar:
    st.header("Debug Panel")
    if st.session_state.weather_api_status == "Success":
        st.success("OpenWeatherMap API: Connected Successfully")
    else:
        st.warning("OpenWeatherMap API: Not used or Failed")
        st.info("Using OpenAI API Only")
        if st.session_state.weather_api_status:
             st.caption(f"Last Status: {st.session_state.weather_api_status}")

# Define the tool for OpenAI
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather for a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. Syracuse, NY",
                    },
                },
                "required": ["location"],
            },
        },
    }
]

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    if message.get("role") != "system": # Don't show system messages
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Where are you properly?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call OpenAI with tools
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Prepare messages, including system prompt for behavior
        messages_for_api = [
            {"role": "system", "content": "You are a helpful assistant that suggests what to wear and outdoor activities based on the weather. If the user asks for weather advice but does not specify a location, call the get_current_weather function with 'Syracuse, NY'. Always provide the weather information first, then your suggestions."},
        ] + [m for m in st.session_state.messages if m.get("role") != "system"]

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages_for_api,
                tools=tools,
                tool_choice="auto",
            )
            
            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls
            
            if tool_calls:
                available_functions = {
                    "get_current_weather": get_current_weather,
                }
                messages_for_api.append(response_message)
                
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_to_call = available_functions[function_name]
                    function_args = json.loads(tool_call.function.arguments)
                    location = function_args.get("location", "Syracuse, NY")
                    
                    function_response = function_to_call(location=location)
                    
                    messages_for_api.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": function_response,
                        }
                    )
                
                # Second call to get the final response
                second_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages_for_api,
                )
                full_response = second_response.choices[0].message.content
            else:
                full_response = response_message.content
                
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
