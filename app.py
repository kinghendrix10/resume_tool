import streamlit as st
from resume_tool import resume_tool
import os

def main():
    st.title("Resume Analysis Tool")

    uploaded_files = st.file_uploader("Choose PDF, DOC, or DOCX files", type=["pdf", "doc", "docx"], accept_multiple_files=True)

    if uploaded_files:
        file_paths = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join("tempDir", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(file_path)

        result = resume_tool(file_paths)
        st.write(result[0])

if __name__ == "__main__":
    main()
