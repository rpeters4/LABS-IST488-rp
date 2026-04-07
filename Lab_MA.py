import os
import json
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Lab MA: Multi-Agent Trip Planner                                          ║
# ║  Build a multi-agent system using LangChain & LangGraph                    ║
# ║                                                                            ║
# ║  In this lab you will:                                                     ║
# ║    1. Set up the environment and initialize the LLM                        ║
# ║    2. Define tools and create three specialist agents                      ║
# ║    3. Create a Supervisor agent to orchestrate the specialists             ║
# ║    4. Build a Streamlit interface to interact with the multi-agent system  ║
# ║    5. Compare single-agent vs. multi-agent performance                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: SETUP & ENVIRONMENT (~5 minutes)
# ══════════════════════════════════════════════════════════════════════════════
#
# 📸 SCREENSHOT FOR SLIDES: Take a screenshot of the running app showing the
#    title, description, and the sidebar controls (destination, days, budget
#    level, interests). This demonstrates the initial UI setup.
#
# In this section we:
#   - Load API keys from Streamlit secrets
#   - Initialize the ChatOpenAI model
#   - Set up the page title and description
# ──────────────────────────────────────────────────────────────────────────────

st.title("Lab MA: Multi-Agent Trip Planner ✈️🌍")
st.caption(
    "🤖 Powered by **LangGraph** — a Supervisor agent orchestrates three "
    "specialist agents (Research, Budget, Itinerary) to plan your perfect trip."
)

st.markdown(
    """
    ### How It Works
    This app uses a **multi-agent system** where:
    - A **Supervisor** receives your request and decides which specialist(s) to call
    - A **Research Agent** finds information about your destination
    - A **Budget Agent** estimates costs for your trip
    - An **Itinerary Agent** creates a day-by-day schedule

    No single agent can plan a complete trip — they must **collaborate** through the Supervisor!
    """
)

# --- API Key Setup ---
try:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error(
        "OpenAI API key not found. Please add OPENAI_API_KEY to "
        "`.streamlit/secrets.toml`",
        icon="🗝️",
    )
    st.stop()

# Initialize the LLM
# We use gpt-4o-mini for the agents (cost-effective) and gpt-4o for the
# supervisor (needs stronger reasoning to route tasks correctly).
agent_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
supervisor_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: BUILDING SPECIALIST AGENTS (~10 minutes)
# ══════════════════════════════════════════════════════════════════════════════
#
# 📸 SCREENSHOT FOR SLIDES: After running a query, expand the "Agent Activity
#    Log" in the sidebar. Screenshot the log showing which agents were called
#    and in what order. This demonstrates agent routing and coordination.
#
# In this section we:
#   - Define 3 tool functions using the @tool decorator
#   - Create 3 specialist agents using create_react_agent
#
# Each tool returns simulated (mock) data so the lab works without external
# travel APIs. The LLM still reasons about the data and composes its response.
# ──────────────────────────────────────────────────────────────────────────────

# --- Tool 1: Destination Research ---
@tool
def search_destination(query: str) -> str:
    """Search for travel information about a destination.
    Use this tool to find details about attractions, culture, weather,
    and practical travel tips for any destination."""

    # Simulated travel data — in a production system this would call a real API
    destinations = {
        "paris": {
            "highlights": "Eiffel Tower, Louvre Museum, Notre-Dame, Montmartre, Seine River cruises",
            "best_time": "April-June or September-October for mild weather",
            "culture": "Rich art scene, café culture, fashion capital, French cuisine",
            "tips": "Learn basic French phrases, metro is efficient, museums closed on Tuesdays",
            "weather": "Mild summers (20-25°C), cold winters (3-7°C), rain year-round",
        },
        "tokyo": {
            "highlights": "Shibuya Crossing, Senso-ji Temple, Akihabara, Mt. Fuji day trips, Tsukiji Market",
            "best_time": "March-May (cherry blossoms) or October-November (fall foliage)",
            "culture": "Blend of ancient and ultra-modern, anime culture, tea ceremonies, onsen baths",
            "tips": "Get a Suica card for transit, bow as greeting, cash still common, shoes off indoors",
            "weather": "Hot humid summers, mild winters, rainy season in June",
        },
        "new york": {
            "highlights": "Statue of Liberty, Central Park, Times Square, Broadway, Brooklyn Bridge",
            "best_time": "April-June or September-November for pleasant weather",
            "culture": "Diverse food scene, world-class museums, live music and theater",
            "tips": "Get a MetroCard, walk everywhere in Manhattan, book Broadway tickets early",
            "weather": "Hot summers (28-32°C), cold winters (-2 to 4°C), beautiful fall colors",
        },
        "rome": {
            "highlights": "Colosseum, Vatican Museums, Trevi Fountain, Pantheon, Roman Forum",
            "best_time": "April-May or September-October to avoid summer crowds",
            "culture": "Ancient history everywhere, incredible food, gelato culture, espresso bars",
            "tips": "Wear comfortable shoes, book Vatican tickets in advance, beware of pickpockets",
            "weather": "Hot dry summers (30°C+), mild winters (8-12°C), very pleasant spring/fall",
        },
    }

    # Search for matching destination (case-insensitive)
    query_lower = query.lower()
    for city, info in destinations.items():
        if city in query_lower:
            return json.dumps(
                {"destination": city.title(), **info}, indent=2
            )

    # Fallback for unknown destinations — the LLM will use its own knowledge
    return json.dumps(
        {
            "destination": query,
            "highlights": "Popular local attractions and landmarks",
            "best_time": "Research the specific climate for optimal travel dates",
            "culture": "Unique local culture and traditions to explore",
            "tips": "Check visa requirements and local customs before traveling",
            "weather": "Check a weather service for current conditions",
        },
        indent=2,
    )


# --- Tool 2: Budget Calculator ---
@tool
def calculate_budget(destination: str, days: int, budget_level: str) -> str:
    """Calculate an estimated travel budget for a trip.
    Use this tool when asked about costs, expenses, or budget planning.
    budget_level should be one of: 'budget', 'moderate', or 'luxury'."""

    # Daily cost estimates by budget level (USD)
    cost_table = {
        "budget": {
            "accommodation": 60,
            "food": 30,
            "transport": 15,
            "activities": 20,
            "misc": 10,
        },
        "moderate": {
            "accommodation": 150,
            "food": 60,
            "transport": 30,
            "activities": 50,
            "misc": 25,
        },
        "luxury": {
            "accommodation": 350,
            "food": 120,
            "transport": 60,
            "activities": 100,
            "misc": 50,
        },
    }

    level = budget_level.lower() if budget_level.lower() in cost_table else "moderate"
    daily = cost_table[level]
    daily_total = sum(daily.values())

    # Estimate flights based on destination
    flight_estimates = {
        "paris": 800,
        "tokyo": 1200,
        "new york": 400,
        "rome": 900,
        "default": 700,
    }
    flight_cost = flight_estimates.get(destination.lower(), flight_estimates["default"])

    budget_breakdown = {
        "destination": destination,
        "budget_level": level,
        "duration_days": days,
        "estimated_flights_roundtrip": flight_cost,
        "daily_breakdown": {k: f"${v}/day" for k, v in daily.items()},
        "daily_total": f"${daily_total}/day",
        "total_daily_costs": f"${daily_total * days}",
        "estimated_grand_total": f"${flight_cost + daily_total * days}",
        "money_saving_tips": [
            "Book flights 2-3 months in advance",
            "Use public transit instead of taxis",
            "Eat at local restaurants, not tourist traps",
            "Look for free walking tours and museum free-days",
        ],
    }
    return json.dumps(budget_breakdown, indent=2)


# --- Tool 3: Itinerary Builder ---
@tool
def create_schedule(destination: str, days: int, interests: str) -> str:
    """Create a day-by-day travel itinerary.
    Use this tool when asked to plan a schedule or itinerary.
    interests should be a comma-separated list of traveler interests."""

    interest_list = [i.strip().lower() for i in interests.split(",")]

    # Build a generic but realistic itinerary
    activities_pool = {
        "food": [
            "Food tour of local markets",
            "Cooking class with a local chef",
            "Fine dining at a top-rated restaurant",
            "Street food walking tour",
        ],
        "history": [
            "Guided historical walking tour",
            "Visit to the national museum",
            "Explore ancient ruins or monuments",
            "Architecture and heritage tour",
        ],
        "art": [
            "Visit the main art gallery",
            "Street art and mural walk",
            "Local artisan workshop visit",
            "Contemporary art museum tour",
        ],
        "nature": [
            "Scenic park or garden walk",
            "Day hike at a nearby trail",
            "Sunset viewpoint excursion",
            "Botanical garden visit",
        ],
        "nightlife": [
            "Rooftop bar with city views",
            "Live music venue",
            "Local pub crawl experience",
            "Evening river or harbor cruise",
        ],
        "shopping": [
            "Visit to famous shopping district",
            "Local flea market exploration",
            "Souvenir shopping in artisan quarter",
            "Designer outlet excursion",
        ],
    }

    # Collect relevant activities
    available = []
    for interest in interest_list:
        for key, acts in activities_pool.items():
            if key in interest:
                available.extend(acts)

    # Fallback activities
    if not available:
        available = [
            "Explore the city center on foot",
            "Visit a popular local attraction",
            "Try a local restaurant",
            "Relax at a scenic viewpoint",
            "Visit a museum or gallery",
            "Explore a local neighborhood",
        ]

    # Build the itinerary
    itinerary = {"destination": destination, "total_days": days, "schedule": []}

    for day in range(1, days + 1):
        day_plan = {
            "day": day,
            "morning": available[(day * 3) % len(available)],
            "afternoon": available[(day * 3 + 1) % len(available)],
            "evening": available[(day * 3 + 2) % len(available)],
        }
        itinerary["schedule"].append(day_plan)

    itinerary["general_tips"] = [
        "Start early to beat crowds at popular attractions",
        "Leave buffer time for spontaneous discoveries",
        "Book popular restaurants and tours in advance",
    ]

    return json.dumps(itinerary, indent=2)


# --- Create Specialist Agents ---
# Each agent gets:
#   - A model to reason with
#   - A list of tools it can use
#   - A name (used by the supervisor for routing)
#   - A prompt that defines its personality and expertise

research_agent = create_react_agent(
    model=agent_llm,
    tools=[search_destination],
    name="research_agent",
    prompt=(
        "You are a travel research specialist. Your job is to find detailed "
        "information about travel destinations including attractions, culture, "
        "weather, and practical tips. Always use the search_destination tool "
        "to look up destination information. Present your findings in a clear, "
        "organized format. You MUST call the tool, do not make up information."
    ),
)

budget_agent = create_react_agent(
    model=agent_llm,
    tools=[calculate_budget],
    name="budget_agent",
    prompt=(
        "You are a travel budget specialist. Your job is to estimate trip costs "
        "and provide budget breakdowns. Always use the calculate_budget tool "
        "to generate cost estimates. Include practical money-saving tips. "
        "You MUST call the tool, do not make up numbers."
    ),
)

itinerary_agent = create_react_agent(
    model=agent_llm,
    tools=[create_schedule],
    name="itinerary_agent",
    prompt=(
        "You are a travel itinerary specialist. Your job is to create detailed "
        "day-by-day travel schedules. Always use the create_schedule tool to "
        "generate itineraries. Tailor the schedule to the traveler's interests. "
        "You MUST call the tool, do not make up schedules."
    ),
)


# ══════════════════════════════════════════════════════════════════════════════
# PART 3: CREATING THE SUPERVISOR (~8 minutes)
# ══════════════════════════════════════════════════════════════════════════════
#
# 📸 SCREENSHOT FOR SLIDES: Screenshot the full trip plan output after
#    clicking "Plan My Trip" — showing the Research, Budget, and Itinerary
#    sections combined. This demonstrates how the Supervisor synthesizes
#    results from all three agents into one cohesive plan.
#
# The Supervisor pattern:
#   - Receives the user's request
#   - Decides which specialist agent(s) to delegate to
#   - Routes the task(s) to the appropriate agent(s)
#   - Synthesizes the final response from all agents' outputs
#
# This is the ORCHESTRATOR/SUBAGENT coordination protocol — one of the key
# concepts from the multi-agent systems presentation.
# ──────────────────────────────────────────────────────────────────────────────

workflow = create_supervisor(
    agents=[research_agent, budget_agent, itinerary_agent],
    model=supervisor_llm,
    prompt=(
        "You are a trip planning supervisor managing three specialist agents:\n"
        "  1. research_agent — finds destination information (attractions, culture, tips)\n"
        "  2. budget_agent — calculates trip costs and budget breakdowns\n"
        "  3. itinerary_agent — creates day-by-day travel schedules\n\n"
        "When a user asks to plan a trip, you MUST delegate to ALL THREE agents "
        "to create a comprehensive plan. Route tasks as follows:\n"
        "  - Destination questions → research_agent\n"
        "  - Cost/budget questions → budget_agent\n"
        "  - Schedule/itinerary questions → itinerary_agent\n"
        "  - Full trip planning → ALL agents (research first, then budget, then itinerary)\n\n"
        "After all agents respond, synthesize their outputs into a single, "
        "well-organized trip plan with clear sections for Research, Budget, "
        "and Itinerary."
    ),
)

# Compile the graph — this creates the runnable multi-agent system
multi_agent_app = workflow.compile()


# ══════════════════════════════════════════════════════════════════════════════
# PART 4: STREAMLIT INTERFACE (~10 minutes)
# ══════════════════════════════════════════════════════════════════════════════
#
# 📸 SCREENSHOT FOR SLIDES: Take a screenshot showing the sidebar with all
#    the trip configuration options filled in (destination, days, budget level,
#    interests) alongside the main content area. This shows the complete UI
#    that students build in Part 4.
#
# In this section we:
#   - Create sidebar controls for trip configuration
#   - Build the main interface with "Plan My Trip" button
#   - Invoke the multi-agent graph and display results
#   - Show agent activity in the debug panel
# ──────────────────────────────────────────────────────────────────────────────

# --- Sidebar Controls ---
with st.sidebar:
    st.header("🗺️ Trip Configuration")

    destination = st.text_input(
        "Destination",
        value="Paris",
        placeholder="e.g. Paris, Tokyo, New York, Rome",
    )

    days = st.slider("Trip Duration (days)", min_value=1, max_value=14, value=5)

    budget_level = st.selectbox(
        "Budget Level", options=["Budget", "Moderate", "Luxury"], index=1
    )

    interests = st.multiselect(
        "Your Interests",
        options=["Food", "History", "Art", "Nature", "Nightlife", "Shopping"],
        default=["Food", "History"],
    )

    st.divider()

# --- Session State ---
if "ma_result" not in st.session_state:
    st.session_state.ma_result = None
if "ma_messages" not in st.session_state:
    st.session_state.ma_messages = None
if "ma_single_result" not in st.session_state:
    st.session_state.ma_single_result = None

# --- Main Interface ---
st.divider()

# Build the user query from sidebar inputs
interests_str = ", ".join(interests) if interests else "general sightseeing"
trip_query = (
    f"Plan a {days}-day trip to {destination}. "
    f"My budget level is {budget_level.lower()}. "
    f"My interests include: {interests_str}. "
    f"Please provide destination research, a budget breakdown, "
    f"and a day-by-day itinerary."
)

# Show the constructed query
with st.expander("📝 View Generated Query", expanded=False):
    st.code(trip_query, language=None)

# Plan My Trip button
if st.button("✈️ Plan My Trip", type="primary", use_container_width=True):
    with st.spinner("🤖 Supervisor is coordinating agents... This may take a moment."):
        try:
            result = multi_agent_app.invoke(
                {"messages": [{"role": "user", "content": trip_query}]}
            )
            st.session_state.ma_result = result["messages"][-1].content
            st.session_state.ma_messages = result["messages"]
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.session_state.ma_result = None

# Display results
if st.session_state.ma_result:
    st.subheader("🌟 Your Trip Plan")
    st.markdown(st.session_state.ma_result)

    #
    # 📸 SCREENSHOT FOR SLIDES: Expand the "Agent Activity Log" below and
    #    screenshot it. This shows students the full message trace — which
    #    agents were called, their tool calls, and the order of execution.
    #    This is a KEY visual for explaining multi-agent coordination.
    #

    # --- Agent Activity Log (Debug Panel) ---
    if st.session_state.ma_messages:
        with st.sidebar:
            st.header("🔍 Agent Activity Log")
            agent_calls = []

            for msg in st.session_state.ma_messages:
                # Detect agent messages by checking the 'name' attribute
                msg_name = getattr(msg, "name", None)
                msg_type = type(msg).__name__

                if msg_name and msg_name in [
                    "research_agent",
                    "budget_agent",
                    "itinerary_agent",
                    "supervisor",
                ]:
                    agent_calls.append(msg_name)

                # Also detect tool calls
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    for tc in tool_calls:
                        tool_name = tc.get("name", "unknown") if isinstance(tc, dict) else getattr(tc, "name", "unknown")
                        agent_calls.append(f"🔧 Tool: {tool_name}")

            if agent_calls:
                st.write("**Execution Order:**")
                for i, agent in enumerate(agent_calls, 1):
                    if agent.startswith("🔧"):
                        st.write(f"  {i}. {agent}")
                    else:
                        emoji = {
                            "supervisor": "👔",
                            "research_agent": "🔬",
                            "budget_agent": "💰",
                            "itinerary_agent": "📅",
                        }.get(agent, "🤖")
                        st.write(f"  {i}. {emoji} {agent}")

            st.write(f"**Total messages:** {len(st.session_state.ma_messages)}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 5: SINGLE-AGENT COMPARISON & REFLECTION (~7 minutes)
# ══════════════════════════════════════════════════════════════════════════════
#
# 📸 SCREENSHOT FOR SLIDES: Take a screenshot showing BOTH the multi-agent
#    result (above) and the single-agent result (below) side by side. This
#    is the most important screenshot — it visually demonstrates WHY
#    multi-agent systems outperform single-agent designs. Highlight the
#    differences in depth, structure, and accuracy between the two.
#
# In this section we:
#   - Add a "Compare with Single Agent" mode
#   - Send the SAME query to a single LLM (no tools, no agents)
#   - Students observe the difference in response quality
#
# This directly connects to the presentation topic: "How do agents in a
# MAS show adaptability and improved performance compared to a single agent?"
# ──────────────────────────────────────────────────────────────────────────────

st.divider()
st.subheader("🔬 Experiment: Single Agent vs. Multi-Agent")
st.markdown(
    """
    **How does a single LLM compare to our multi-agent system?**

    Click below to send the *exact same query* to a single GPT-4o-mini
    (with no tools and no agent coordination). Compare the depth, accuracy,
    and structure of the response.
    """
)

if st.button("🧪 Run Single-Agent Comparison", use_container_width=True):
    with st.spinner("Asking a single LLM (no agents, no tools)..."):
        try:
            single_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            single_result = single_llm.invoke(trip_query)
            st.session_state.ma_single_result = single_result.content
        except Exception as e:
            st.error(f"Error: {e}")

if st.session_state.ma_single_result:
    st.markdown("### Single-Agent Response")
    st.markdown(st.session_state.ma_single_result)

    #
    # 📸 SCREENSHOT FOR SLIDES: Screenshot this reflection section with
    #    both results visible. Use this to discuss:
    #    1. The multi-agent response has REAL data (from tools)
    #    2. The single-agent response is generic (no tool access)
    #    3. The multi-agent response is more structured (specialized agents)
    #    4. The supervisor successfully coordinated the workflow
    #

    st.divider()
    st.markdown(
        """
        ### 🤔 Reflection Questions
        Compare the two responses above and consider:

        1. **Data Quality**: Which response has more specific, actionable information?
           (Hint: the multi-agent system had access to tools with real data)

        2. **Structure**: Which response is better organized? Why?
           (Hint: each specialist agent focused on one aspect of the plan)

        3. **Coordination**: How did the Supervisor decide which agents to call?
           (Check the Agent Activity Log in the sidebar)

        4. **Trade-offs**: What are the downsides of a multi-agent approach?
           (Think about: latency, cost, complexity, failure modes)
        """
    )
