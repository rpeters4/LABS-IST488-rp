import streamlit as st

# Define pages
lab1 = st.Page("Lab1.py", title="Lab 1", icon=":material/description:")
lab2 = st.Page("Lab2.py", title="Lab 2", icon=":material/summarize:", default=True)

# Create navigation
pg = st.navigation([lab1, lab2])

# Configure page
st.set_page_config(page_title="IST 488 Labs", page_icon=":material/school:")

# Run the selected page
pg.run()
