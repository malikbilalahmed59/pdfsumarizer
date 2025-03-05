from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from authlib.integrations.starlette_client import OAuth, OAuthError
from starlette.middleware.sessions import SessionMiddleware
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from cachetools import TTLCache
from dotenv import load_dotenv
import os
import base64
import openai
import PyPDF2
import io
import httpx
import asyncio

# Load environment variables
load_dotenv()
CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')
OPENAI_APIKEY = os.getenv('OPENAI_APIKEY')
SECRET_KEY = os.getenv('SECRET_KEY')

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

# Configure Jinja templates
templates = Jinja2Templates(directory="static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup Google OAuth
oauth = OAuth()
oauth.register(
    name='google',
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid email profile https://www.googleapis.com/auth/gmail.readonly',
        'access_type': 'offline',
        'prompt': 'consent'
    }
)
openai.api_key = OPENAI_APIKEY

# Cache for storing email search results (reduces API calls)
cache = TTLCache(maxsize=50, ttl=600)  # Store up to 50 results for 10 minutes


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """ Home Page: Redirects user to login if not authenticated """
    user = request.session.get('user')
    if user:
        return templates.TemplateResponse("home.html", {"request": request, "user": user})
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/login")
async def login(request: Request):
    """ Redirects users to Google OAuth login """
    redirect_uri = request.url_for('auth')
    return await oauth.google.authorize_redirect(request, redirect_uri)


@app.get("/auth")
async def auth(request: Request):
    """ Handles Google OAuth authentication """
    try:
        token = await oauth.google.authorize_access_token(request)
        user = token.get('userinfo', await oauth.google.userinfo(token=token))
        request.session['user'] = dict(user)
        request.session['token'] = dict(token)
        return RedirectResponse(url='/')
    except OAuthError as error:
        return HTMLResponse(f"<h1>{error.error}</h1>")


@app.get("/logout")
async def logout(request: Request):
    """ Logs out the user and clears session """
    request.session.clear()
    return RedirectResponse(url='/')


@app.get("/search")
async def search(request: Request, query: str, background_tasks: BackgroundTasks):
    """ Searches for PDF attachments in Gmail and analyzes them asynchronously """
    user = request.session.get('user')
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    # Use cached results if available
    if query in cache:
        return JSONResponse(content=cache[query])

    # Fetch emails asynchronously
    results = await fetch_emails(request)

    # Process PDFs in background (so Vercel doesn’t timeout)
    background_tasks.add_task(process_pdfs, results, query, request)

    return JSONResponse({"message": "Processing in background. Check back later."})


async def fetch_emails(request: Request):
    """ Asynchronous function to fetch email messages from Gmail API """
    token = request.session.get('token')
    credentials = refresh_token_if_needed(token)

    gmail_service = build('gmail', 'v1', credentials=credentials)

    search_query = "has:attachment filename:pdf"
    results = gmail_service.users().messages().list(userId='me', q=search_query, maxResults=5).execute()

    return results.get('messages', [])


def refresh_token_if_needed(token):
    """ Refresh Google OAuth token if expired """
    credentials = Credentials(
        token=token['access_token'],
        refresh_token=token.get('refresh_token'),
        token_uri='https://oauth2.googleapis.com/token',
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET
    )

    if credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())

    return credentials


async def process_pdfs(messages, query, request):
    """ Process PDF attachments asynchronously and analyze them with GPT """
    token = request.session.get('token')
    credentials = refresh_token_if_needed(token)

    gmail_service = build('gmail', 'v1', credentials=credentials)
    pdf_list = []

    for message in messages:
        msg = gmail_service.users().messages().get(userId='me', id=message['id']).execute()
        subject = next(header['value'] for header in msg['payload']['headers'] if header['name'] == 'Subject')
        attachments = msg['payload'].get('parts', [])

        for attachment in attachments:
            if attachment.get('filename', '').endswith('.pdf'):
                attachment_id = attachment['body']['attachmentId']
                attachment_data = gmail_service.users().messages().attachments().get(userId='me',
                                                                                     messageId=message['id'],
                                                                                     id=attachment_id).execute()
                pdf_data = base64.urlsafe_b64decode(attachment_data['data'].encode('UTF-8'))

                pdf_text = extract_pdf_first_100_sentences(pdf_data)
                gpt_response = await analyze_pdf_with_gpt(pdf_text, query)

                pdf_list.append({
                    "id": message['id'],
                    "subject": subject,
                    "attachment": attachment['filename'],
                    "gpt_summary": gpt_response
                })

    # Cache the results to prevent repeated slow processing
    cache[query] = pdf_list


def extract_pdf_first_100_sentences(pdf_data: bytes) -> str:
    """ Extracts the first 100 sentences from a PDF file """
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_data))
        text = " ".join([page.extract_text() or "" for page in pdf_reader.pages])
        sentences = text.split('.')
        return '. '.join(sentences[:100])  # Return first 100 sentences
    except Exception as e:
        return f"Error processing PDF: {e}"


async def analyze_pdf_with_gpt(pdf_content: str, query: str) -> str:
    """ Uses OpenAI GPT to analyze PDF content asynchronously """
    try:
        prompt = f"Analyze the following content based on query '{query}'. Provide a summary:\n\n{pdf_content}"
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_APIKEY}"},
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "system", "content": "You are an expert assistant."},
                                 {"role": "user", "content": prompt}],
                    "max_tokens": 500
                }
            )
            return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error analyzing PDF: {e}"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
