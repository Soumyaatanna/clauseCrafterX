import requests
import PyPDF2
import docx
import io

def extract_text_from_url(url: str) -> str:
    try:
        response = requests.get(url)
        response.raise_for_status()
        file_extension = url.split('?')[0].split('.')[-1].lower()
        file_content = io.BytesIO(response.content)

        if file_extension == 'pdf':
            pdf_reader = PyPDF2.PdfReader(file_content)
            text = "".join(page.extract_text() for page in pdf_reader.pages)
        elif file_extension == 'docx':
            doc = docx.Document(file_content)
            text = "\n".join([para.text for para in doc.paragraphs])
        else:
            text = response.text
        return text
    except Exception as e:
        raise Exception(f"Failed to parse document: {e}")