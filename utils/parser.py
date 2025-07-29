import requests
import fitz  # PyMuPDF
import docx
import io

def extract_text_from_url(url: str) -> str:
    """Downloads and extracts text from a PDF, DOCX, or plain text file at the given URL."""
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()

        file_extension = url.split('?')[0].split('.')[-1].lower()
        file_content = io.BytesIO(response.content)

        if file_extension == 'pdf':
            text = ""
            with fitz.open(stream=file_content, filetype="pdf") as doc:
                for page in doc:
                    text += page.get_text() + "\n"
        elif file_extension == 'docx':
            doc = docx.Document(file_content)
            text = "\n".join([para.text.strip() for para in doc.paragraphs if para.text.strip()])
        elif file_extension == 'txt':
            text = file_content.read().decode("utf-8")
        else:
            raise ValueError(f"Unsupported file type: .{file_extension}")

        return text.strip()

    except requests.exceptions.RequestException as req_err:
        raise Exception(f"Network error while downloading document: {req_err}")
    except Exception as parse_err:
        raise Exception(f"Failed to parse document: {parse_err}")