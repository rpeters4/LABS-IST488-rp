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
# In this section we:
#   - Load API keys from Streamlit secrets
#   - Initialize the ChatOpenAI model
#   - Set up the page title and description
# ──────────────────────────────────────────────────────────────────────────────

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ 📸 BEGIN SCREENSHOT 1 — "App Overview & Sidebar Controls"                  │
# │ Screenshot the running app showing: the title, the "How It Works"          │
# │ description, AND the sidebar with Trip Configuration controls              │
# │ (Destination, Trip Duration slider, Budget Level, Your Interests).         │
# │ This slide demonstrates the initial UI and the user-facing interface.      │
# └─────────────────────────────────────────────────────────────────────────────┘

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

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ 📸 END SCREENSHOT 1                                                        │
# └─────────────────────────────────────────────────────────────────────────────┘


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: BUILDING SPECIALIST AGENTS (~10 minutes)
# ══════════════════════════════════════════════════════════════════════════════
#
# In this section we:
#   - Define 3 tool functions using the @tool decorator
#   - Create 3 specialist agents using create_react_agent
#
# Each tool returns simulated (mock) data so the lab works without external
# travel APIs. The LLM still reasons about the data and composes its response.
# ──────────────────────────────────────────────────────────────────────────────

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ 📸 BEGIN SCREENSHOT 2 — "Tool Definitions & Agent Creation"                │
# │ Screenshot the CODE in your editor showing:                                │
# │   • The @tool decorated functions (search_destination, calculate_budget,   │
# │     create_schedule)                                                       │
# │   • The create_react_agent() calls that wire each tool to an agent         │
# │ This slide shows how tools and agents are defined in LangGraph.            │
# └─────────────────────────────────────────────────────────────────────────────┘

# --- Load travel data from JSON ---
# All destination info, costs, and activities are stored in travel_data.json
# This keeps the data separate from the logic and makes it easy to extend.
with open("travel_data.json", "r") as f:
    TRAVEL_DATA = json.load(f)

# --- Tool 1: Destination Research ---
@tool
def search_destination(query: str) -> str:
    """Search for travel information about a destination.
    Use this tool to find details about attractions, culture, weather,
    and practical travel tips for any destination."""

    destinations = TRAVEL_DATA["destinations"]

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

    cost_table = TRAVEL_DATA["daily_costs"]
    flight_estimates = TRAVEL_DATA["flight_estimates"]

    level = budget_level.lower() if budget_level.lower() in cost_table else "moderate"
    daily = cost_table[level]
    daily_total = sum(daily.values())

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
        "money_saving_tips": TRAVEL_DATA["money_saving_tips"],
    }
    return json.dumps(budget_breakdown, indent=2)


# --- Tool 3: Itinerary Builder ---
@tool
def create_schedule(destination: str, days: int, interests: str) -> str:
    """Create a day-by-day travel itinerary.
    Use this tool when asked to plan a schedule or itinerary.
    interests should be a comma-separated list of traveler interests."""

    interest_list = [i.strip().lower() for i in interests.split(",")]
    activities_pool = TRAVEL_DATA["activities"]

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

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ 📸 END SCREENSHOT 2                                                        │
# └─────────────────────────────────────────────────────────────────────────────┘


# ══════════════════════════════════════════════════════════════════════════════
# PART 3: CREATING THE SUPERVISOR (~8 minutes)
# ══════════════════════════════════════════════════════════════════════════════
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

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ 📸 BEGIN SCREENSHOT 3 — "Supervisor Wiring & Compiled Graph"               │
# │ Screenshot the CODE showing:                                               │
# │   • The create_supervisor() call with the three agents and prompt          │
# │   • The workflow.compile() call                                            │
# │ This slide explains how the orchestrator is configured and compiled.       │
# └─────────────────────────────────────────────────────────────────────────────┘

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

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ 📸 END SCREENSHOT 3                                                        │
# └─────────────────────────────────────────────────────────────────────────────┘


# ══════════════════════════════════════════════════════════════════════════════
# PART 4: STREAMLIT INTERFACE (~10 minutes)
# ══════════════════════════════════════════════════════════════════════════════
#
# In this section we:
#   - Create sidebar controls for trip configuration
#   - Build the main interface with "Plan My Trip" button
#   - Invoke the multi-agent graph and display results
#   - Show agent activity in the debug panel
# ──────────────────────────────────────────────────────────────────────────────

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ 📸 BEGIN SCREENSHOT 4 — "Full Trip Plan Output + Agent Activity Log"       │
# │ After clicking "Plan My Trip", screenshot the RUNNING APP showing:         │
# │   • The "Your Trip Plan" section with Research, Budget & Itinerary output  │
# │   • The sidebar "Agent Activity Log" expanded (showing execution order)    │
# │ This slide demonstrates the Supervisor synthesizing all three agents'      │
# │ outputs AND the agent routing/coordination in the debug log.               │
# └─────────────────────────────────────────────────────────────────────────────┘

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

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ 📸 END SCREENSHOT 4                                                        │
# └─────────────────────────────────────────────────────────────────────────────┘


# ══════════════════════════════════════════════════════════════════════════════
# PART 5: SINGLE-AGENT COMPARISON & REFLECTION (~7 minutes)
# ══════════════════════════════════════════════════════════════════════════════
#
# In this section we:
#   - Add a "Compare with Single Agent" mode
#   - Send the SAME query to a single LLM (no tools, no agents)
#   - Students observe the difference in response quality
#
# This directly connects to the presentation topic: "How do agents in a
# MAS show adaptability and improved performance compared to a single agent?"
# ──────────────────────────────────────────────────────────────────────────────

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ 📸 BEGIN SCREENSHOT 5 — "Single-Agent vs Multi-Agent Comparison"           │
# │ Screenshot the RUNNING APP showing BOTH results visible:                   │
# │   • The multi-agent "Your Trip Plan" output (from Part 4 above)            │
# │   • The single-agent response (below, after clicking the button)           │
# │ This is the MOST IMPORTANT slide — it visually demonstrates WHY            │
# │ multi-agent systems outperform single-agent designs. Highlight the         │
# │ differences in depth, structure, and tool-backed accuracy.                 │
# └─────────────────────────────────────────────────────────────────────────────┘

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

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ 📸 END SCREENSHOT 5                                                        │
# └─────────────────────────────────────────────────────────────────────────────┘


# ══════════════════════════════════════════════════════════════════════════════
# PART 6: CHATBOT MODE — USING MessagesState & StateGraph
# ══════════════════════════════════════════════════════════════════════════════
#
# In Parts 1-5 we used create_supervisor() which is a HIGH-LEVEL helper.
# Under the hood, it builds a StateGraph for you. In this section, we
# build a conversational chatbot using the LOWER-LEVEL LangGraph primitives:
#
#   - MessagesState: a TypedDict that holds a list of messages as graph state
#   - StateGraph: the core graph builder where you add nodes and edges
#   - Nodes: functions that process the current state and return updates
#   - Edges: connections between nodes (static or conditional)
#
# This gives you a deeper understanding of how multi-agent systems work
# under the hood, and lets you build a CONVERSATIONAL version that
# remembers previous messages.
# ──────────────────────────────────────────────────────────────────────────────

from langgraph.graph import StateGraph, MessagesState, START, END

st.divider()
st.subheader("💬 Part 6: Multi-Agent Chatbot")
st.markdown(
    """
    This chatbot uses **MessagesState** and **StateGraph** — the lower-level
    LangGraph primitives — to build a conversational multi-agent system.
    Instead of the high-level `create_supervisor()`, the graph is wired
    manually with nodes and edges.
    """
)

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ 📸 BEGIN SCREENSHOT 6 — "Chatbot with MessagesState & StateGraph"          │
# │ Screenshot the RUNNING APP showing the chatbot in action:                  │
# │   • A user message and a multi-agent response in the chat history          │
# │   • The sidebar showing which agents handled the latest message            │
# │ This slide demonstrates conversational multi-agent coordination.           │
# └─────────────────────────────────────────────────────────────────────────────┘


# --- Define the graph nodes ---
# Each node is a function that takes the current state (MessagesState)
# and returns a dict with updated messages.

def supervisor_node(state: MessagesState):
    """The supervisor reads the latest user message and decides which
    agent(s) to route to. It returns routing instructions as a message."""
    messages = state["messages"]

    routing_prompt = (
        "You are a trip planning supervisor. Based on the user's message, "
        "decide which specialist to call:\n"
        "  - 'research' for destination info questions\n"
        "  - 'budget' for cost/money questions\n"
        "  - 'itinerary' for schedule/planning questions\n"
        "  - 'all' if the user wants a complete trip plan\n"
        "  - 'chat' for general conversation or follow-ups\n\n"
        "Respond with ONLY one word: research, budget, itinerary, all, or chat."
    )

    response = supervisor_llm.invoke(
        [{"role": "system", "content": routing_prompt}] + messages
    )
    return {"messages": [response]}


def research_node(state: MessagesState):
    """Runs the research agent on the conversation."""
    result = research_agent.invoke({"messages": state["messages"]})
    return {"messages": result["messages"]}


def budget_node(state: MessagesState):
    """Runs the budget agent on the conversation."""
    result = budget_agent.invoke({"messages": state["messages"]})
    return {"messages": result["messages"]}


def itinerary_node(state: MessagesState):
    """Runs the itinerary agent on the conversation."""
    result = itinerary_agent.invoke({"messages": state["messages"]})
    return {"messages": result["messages"]}


def synthesizer_node(state: MessagesState):
    """Reads all agent outputs and produces a final conversational response."""
    messages = state["messages"]

    synth_prompt = (
        "You are a friendly trip planning assistant. Based on the full "
        "conversation above (which may include outputs from specialist agents), "
        "provide a helpful, well-organized response to the user. "
        "Be conversational — the user may ask follow-up questions."
    )

    response = agent_llm.invoke(
        [{"role": "system", "content": synth_prompt}] + messages
    )
    return {"messages": [response]}


# --- Route function for conditional edges ---
def route_from_supervisor(state: MessagesState):
    """Reads the supervisor's routing decision and returns the next node name."""
    last_msg = state["messages"][-1].content.strip().lower()

    if "research" in last_msg:
        return "research"
    elif "budget" in last_msg:
        return "budget"
    elif "itinerary" in last_msg:
        return "itinerary"
    elif "all" in last_msg:
        return "all"
    else:
        return "chat"


def run_all_agents(state: MessagesState):
    """Sequentially runs all three agents for a full trip plan."""
    research_result = research_agent.invoke({"messages": state["messages"]})
    budget_result = budget_agent.invoke({"messages": state["messages"]})
    itinerary_result = itinerary_agent.invoke({"messages": state["messages"]})

    # Combine all new messages
    all_new = (
        research_result["messages"] +
        budget_result["messages"] +
        itinerary_result["messages"]
    )
    return {"messages": all_new}


# --- Build the StateGraph ---
# This is the manual equivalent of create_supervisor().
# We define nodes, then connect them with edges.

chatbot_graph = StateGraph(MessagesState)

# Add nodes
chatbot_graph.add_node("supervisor", supervisor_node)
chatbot_graph.add_node("research", research_node)
chatbot_graph.add_node("budget", budget_node)
chatbot_graph.add_node("itinerary", itinerary_node)
chatbot_graph.add_node("all_agents", run_all_agents)
chatbot_graph.add_node("synthesizer", synthesizer_node)

# Add edges:
# START -> supervisor (every message goes to supervisor first)
chatbot_graph.add_edge(START, "supervisor")

# supervisor -> conditional routing based on the routing decision
chatbot_graph.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {
        "research": "research",
        "budget": "budget",
        "itinerary": "itinerary",
        "all": "all_agents",
        "chat": "synthesizer",
    },
)

# Each specialist -> synthesizer (to produce a final response)
chatbot_graph.add_edge("research", "synthesizer")
chatbot_graph.add_edge("budget", "synthesizer")
chatbot_graph.add_edge("itinerary", "synthesizer")
chatbot_graph.add_edge("all_agents", "synthesizer")

# synthesizer -> END
chatbot_graph.add_edge("synthesizer", END)

# Compile the graph
chatbot_app = chatbot_graph.compile()


# --- Streamlit Chat Interface ---
if "chatbot_messages" not in st.session_state:
    st.session_state.chatbot_messages = []

# Display chat history
for message in st.session_state.chatbot_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if user_input := st.chat_input("Ask me to plan a trip, check costs, or find info..."):
    # Add user message to display history
    st.session_state.chatbot_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build the full message list for the graph
    graph_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.chatbot_messages
    ]

    with st.spinner("🤖 Agents are working..."):
        try:
            result = chatbot_app.invoke({"messages": graph_messages})
            # Get the last AI message as the response
            assistant_response = result["messages"][-1].content

            # Display and store
            with st.chat_message("assistant"):
                st.markdown(assistant_response)
            st.session_state.chatbot_messages.append(
                {"role": "assistant", "content": assistant_response}
            )
        except Exception as e:
            st.error(f"Error: {e}")

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ 📸 END SCREENSHOT 6                                                        │
# └─────────────────────────────────────────────────────────────────────────────┘
