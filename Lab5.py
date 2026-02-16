import streamlit as st
from openai import OpenAI
import requests
import json

# ---------- Page Setup ----------
st.title("Lab 5: What to Wear Bot :partly_sunny:")
st.write("Enter a city name below to get weather-based clothing and activity suggestions.")

# ---------- API Keys ----------
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    openweathermap_api_key = st.secrets["OPENWEATHERMAP_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("API keys not found. Please configure them in .streamlit/secrets.toml")
    st.stop()

client = OpenAI(api_key=openai_api_key)

# ---------- Session State ----------
if "weather_api_status" not in st.session_state:
    st.session_state.weather_api_status = None
if "last_weather_data" not in st.session_state:
    st.session_state.last_weather_data = None

# ---------- Weather Function (Part A) ----------
def get_current_weather(location):
    """
    Get the current weather for a given location using the OpenWeatherMap API.
    Returns a JSON string with temperature, description, humidity, and wind speed.
    """
    # Clean up location string for the API
    location_query = location.strip().replace(", ", ",")
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?q={location_query}&appid={openweathermap_api_key}&units=imperial"
    )

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        weather_info = {
            "location": data.get("name", location),
            "temperature_f": data["main"]["temp"],
            "feels_like_f": data["main"]["feels_like"],
            "description": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"],
            "wind_speed_mph": data["wind"]["speed"],
        }
        st.session_state.weather_api_status = "success"
        st.session_state.last_weather_data = weather_info
        return json.dumps(weather_info)

    except requests.exceptions.HTTPError as e:
        detail = ""
        if e.response is not None:
            detail = f" | Status {e.response.status_code}: {e.response.text}"
        st.session_state.weather_api_status = f"error{detail}"
        return json.dumps({"error": f"HTTP error for '{location}'{detail}"})

    except requests.exceptions.RequestException as e:
        st.session_state.weather_api_status = f"error | {e}"
        return json.dumps({"error": str(e)})

    except KeyError:
        st.session_state.weather_api_status = "error | Unexpected response format"
        return json.dumps({"error": "Location not found or unexpected response format."})


# ---------- OpenAI Tool Definition (Part B, Step 7) ----------
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state/country, e.g. 'Syracuse,NY,US' or 'Lima,Peru'",
                    },
                },
                "required": ["location"],
            },
        },
    }
]

# ---------- User Input ----------
city_input = st.text_input(
    "Enter a city name:",
    placeholder="e.g. Syracuse, NY, US  or  Lima, Peru",
)

if st.button("Get Suggestions", type="primary"):
    # Default to Syracuse, NY if no location provided (Part B, Step 7b)
    city = city_input.strip() if city_input.strip() else "Syracuse, NY"

    with st.spinner("Thinking..."):
        # --- First OpenAI call: let the model decide to call the tool ---
        user_message = f"What should I wear and what outdoor activities can I do today in {city}?"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that suggests appropriate clothing "
                    "and outdoor activities based on the current weather. "
                    "Always call the get_current_weather tool to retrieve real-time "
                    "weather data before making suggestions. "
                    "If no location is provided, default to 'Syracuse,NY'."
                ),
            },
            {"role": "user", "content": user_message},
        ]

        try:
            first_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )

            response_message = first_response.choices[0].message
            tool_calls = response_message.tool_calls

            if tool_calls:
                # The model wants to call get_current_weather
                messages.append(response_message)

                for tool_call in tool_calls:
                    fn_name = tool_call.function.name
                    fn_args = json.loads(tool_call.function.arguments)
                    location_arg = fn_args.get("location", "Syracuse,NY")

                    # Execute the weather function
                    weather_result = get_current_weather(location=location_arg)

                    messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": fn_name,
                            "content": weather_result,
                        }
                    )

                # --- Second OpenAI call: generate clothing / activity advice ---
                second_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                )
                final_answer = second_response.choices[0].message.content
            else:
                # Model answered without calling the tool
                final_answer = response_message.content

            # --- Display the result ---
            st.markdown("---")
            st.subheader(f"Suggestions for {city}")
            st.markdown(final_answer)

        except Exception as e:
            st.error(f"An error occurred: {e}")

# ---------- Debug Sidebar ----------
with st.sidebar:
    st.header("Debug Panel")

    if st.button("Test Weather API"):
        with st.spinner("Testing..."):
            get_current_weather("Syracuse,NY,US")

    status = st.session_state.weather_api_status
    if status == "success":
        st.success("✅ OpenWeatherMap API: Connected")
        if st.session_state.last_weather_data:
            st.json(st.session_state.last_weather_data)
    elif status and status.startswith("error"):
        st.error("❌ OpenWeatherMap API: Failed")
        st.caption(status)
    else:
        st.info("No API call made yet.")
