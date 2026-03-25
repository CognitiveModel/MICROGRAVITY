import time
import google.generativeai as genai

try:
    from coding_agent.utils.config import GEMINI_API_KEY
except ImportError:
    from utils.config import GEMINI_API_KEY

def init_gemini():
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel('models/gemini-2.0-flash')

def get_gemini_response(model, contents, retries=3, backoff=5):
    """
    Retrieves Gemini response with exponential backoff for rate limits.
    Supports list of contents for multimodal (text + images).
    """
    for attempt in range(retries):
        try:
            response = model.generate_content(contents)
            return response.text
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "Resource has been exhausted" in err_msg:
                wait_time = backoff * (2 ** attempt)
                print(f"[GEMINI] Rate limit hit. Waiting {wait_time}s (Attempt {attempt+1}/{retries})...")
                time.sleep(wait_time)
                continue
            return f"ERROR: Gemini API error: {e}"

    return "ERROR: Gemini API rate limit exceeded after retries."
