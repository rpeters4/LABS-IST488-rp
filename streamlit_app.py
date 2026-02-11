import streamlit as st

# Define pages
lab1 = st.Page("Lab1.py", title="Lab 1", icon=":material/description:")
lab2 = st.Page("Lab2.py", title="Lab 2", icon=":material/description:")
lab3 = st.Page("Lab3.py", title="Lab 3", icon=":material/description:")
lab4 = st.Page("Lab4.py", title="Lab 4", icon=":material/description:", default=True)

# Create navigation
pg = st.navigation([lab1, lab2, lab3, lab4])

# Configure page
st.set_page_config(page_title="IST 488 Labs", page_icon=":material/school:")

# Run the selected page
pg.run()
