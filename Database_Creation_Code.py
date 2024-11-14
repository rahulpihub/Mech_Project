import streamlit as st
import sqlite3
import os

# Database connection
db_name = "diseal.db"
conn = sqlite3.connect(db_name)
cursor = conn.cursor()

# Create table if it doesn't exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS knowledge_base (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image BLOB NOT NULL,
        pdf BLOB NOT NULL,
        common_pdf BLOB NOT NULL
    )
''')
conn.commit()

# Function to convert file to binary data
def convert_to_binary(file_path):
    with open(file_path, 'rb') as file:
        return file.read()

# Function to insert data into the database
def insert_data(image_blob, pdf_blob, common_pdf_blob):
    cursor.execute("INSERT INTO knowledge_base (image, pdf, common_pdf) VALUES (?, ?, ?)",
                   (image_blob, pdf_blob, common_pdf_blob))
    conn.commit()

# Streamlit app
st.title("Upload Image, Relevant PDF, and Common PDF")

# File uploads
uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png"])
uploaded_pdf = st.file_uploader("Upload Relevant PDF", type=["pdf"])
common_pdf = st.file_uploader("Upload Common PDF", type=["pdf"])

if uploaded_image and uploaded_pdf and common_pdf:
    # Convert uploaded files to binary data
    image_blob = uploaded_image.read()
    pdf_blob = uploaded_pdf.read()
    common_pdf_blob = common_pdf.read()

    # Insert data into the database
    insert_data(image_blob, pdf_blob, common_pdf_blob)
    
    st.success("Files have been successfully uploaded and stored in the database.")

# Close the database connection when the app is stopped
st.write("Data successfully appended to the database.")
if st.button("Close Database Connection"):
    conn.close()
    st.write("Database connection closed.")
