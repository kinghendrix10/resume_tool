# resume_tool

Created a tool to parse CV in pdf format and build a table of the candidates profiles and skills as required by the user.

Skills.csv is where the needed skills can be saved and used to parse CVs or resumes and also represent the skills as a graph

Dump all the resumes into a single folder and copy the file path then paste it on the 'mypath' variable in the 'init' file and sit back.

Working on making this a web application

## Dependencies

- Streamlit
- Llama 3.1 70b models
- Groq API
- PyPDF2
- docx2txt
- textract
- spaCy
- pandas
- matplotlib
- openai

## Usage

1. Install the required dependencies:
   ```bash
   pip install streamlit PyPDF2 docx2txt textract spacy pandas matplotlib openai
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. Upload PDF, DOC, or DOCX files through the Streamlit web interface.

4. View the analysis results and comparison metrics.
