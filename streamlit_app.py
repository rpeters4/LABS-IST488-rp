import streamlit as st

# Define pages
lab1 = st.Page("Lab1.py", title="Lab 1", icon=":material/description:")
lab2 = st.Page("Lab2.py", title="Lab 2", icon=":material/description:")
lab3 = st.Page("Lab3.py", title="Lab 3", icon=":material/description:")
lab4 = st.Page("Lab4.py", title="Lab 4", icon=":material/description:")
lab5 = st.Page("Lab5.py", title="Lab 5", icon=":material/description:")
lab6 = st.Page("Lab6.py", title="Lab 6", icon=":material/description:")
lab6a = st.Page("Lab6a.py", title="Lab 6a", icon=":material/description:")
lab6b = st.Page("Lab6b.py", title="Lab 6b", icon=":material/description:", default=True)

# Create navigation
pg = st.navigation([lab1, lab2, lab3, lab4, lab5, lab6, lab6a, lab6b])

# Configure page
st.set_page_config(page_title="IST 488 Labs", page_icon=":material/school:")

# Run the selected page
pg.run()
