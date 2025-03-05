from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from authlib.integrations.starlette_client import OAuth, OAuthError
from starlette.middleware.sessions import SessionMiddleware
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from dotenv import load_dotenv
import os
import base64
import openai
import PyPDF2
import io

# ----------------------------------------------------------------------------
# 1. LOAD ENV & GLOBAL CACHES
# ----------------------------------------------------------------------------

load_dotenv()
CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')
OPENAI_APIKEY = os.getenv('OPENAI_APIKEY')
SECRET_KEY = os.getenv('SECRET_KEY')

# In-memory caches (demo only; for production, use Redis or DB)
pdf_text_cache = {}    # key: (message_id, filename) -> extracted text
summary_cache = {}     # key: (message_id, filename, query) -> GPT summary

# ----------------------------------------------------------------------------
# 2. FASTAPI SETUP
# ----------------------------------------------------------------------------

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

templates = Jinja2Templates(directory="static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ----------------------------------------------------------------------------
# 3. SETUP GOOGLE OAUTH & OPENAI
# ----------------------------------------------------------------------------

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

# ----------------------------------------------------------------------------
# 4. ROUTES
# ----------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home Page: redirects user to login if not authenticated."""
    user = request.session.get('user')
    if user:
        return templates.TemplateResponse("home.html", {"request": request, "user": user})
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/login")
async def login(request: Request):
    """Redirect users to Google OAuth login."""
    redirect_uri = request.url_for('auth')
    return await oauth.google.authorize_redirect(request, redirect_uri)


@app.get("/auth")
async def auth(request: Request):
    """Handles Google OAuth authentication."""
    try:
        token = await oauth.google.authorize_access_token(request)
        user = token.get('userinfo')
        if not user:
            user = await oauth.google.userinfo(token=token)
        request.session['user'] = dict(user)
        request.session['token'] = dict(token)
        return RedirectResponse(url='/')
    except OAuthError as error:
        return HTMLResponse(f"<h1>{error.error}</h1>")


@app.get("/logout")
async def logout(request: Request):
    """Logs out the user and clears session."""
    request.session.pop('user', None)
    request.session.pop('token', None)
    request.session.pop('search_query', None)
    request.session.pop('pdf_list', None)
    return RedirectResponse(url='/')


@app.get("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    query: str = Query(..., description="Search query"),
    page: int = Query(1, ge=1, description="Page number"),
):
    """
    Searches PDF attachments in Gmail, uses GPT to summarize them (with caching),
    and implements local pagination (5 items per page).
    """
    user = request.session.get('user')
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    existing_query = request.session.get('search_query')
    pdf_list = request.session.get('pdf_list', [])

    # Check if we need a fresh search (new query or no results cached):
    if (not pdf_list) or (existing_query != query):
        print("🔎 Performing a fresh Gmail search + GPT summaries ...")
        token = request.session.get('token')
        credentials = Credentials(
            token=token['access_token'],
            refresh_token=token.get('refresh_token'),
            token_uri='https://oauth2.googleapis.com/token',
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET
        )
        gmail_service = build('gmail', 'v1', credentials=credentials)

        # For better filtering, you can incorporate the user's query:
        # search_query = f"has:attachment filename:pdf {query}"
        search_query = "has:attachment filename:pdf"

        # Retrieve all messages by handling nextPageToken
        all_messages = []
        response = gmail_service.users().messages().list(userId='me', q=search_query).execute()
        while True:
            messages_chunk = response.get('messages', [])
            if messages_chunk:
                all_messages.extend(messages_chunk)

            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break

            response = gmail_service.users().messages().list(
                userId='me', q=search_query, pageToken=next_page_token
            ).execute()

        print(f"📬 Total messages found with PDFs: {len(all_messages)}")

        # Build new PDF list
        new_pdf_list = []
        for message in all_messages:
            msg = gmail_service.users().messages().get(userId='me', id=message['id']).execute()
            subject = next(
                (header['value'] for header in msg['payload']['headers'] if header['name'] == 'Subject'),
                "No Subject"
            )

            parts = msg['payload'].get('parts', [])
            for part in parts:
                filename = part.get('filename', '')
                if filename.endswith('.pdf'):
                    attachment_id = part.get('body', {}).get('attachmentId')
                    if not attachment_id:
                        continue

                    # 1) Get the PDF text, using the local cache if possible
                    pdf_text = get_cached_pdf_text(
                        gmail_service, message['id'], filename, attachment_id
                    )
                    # 2) Summarize with GPT, also caching the result
                    gpt_response = get_cached_gpt_summary(
                        message_id=message['id'],
                        filename=filename,
                        pdf_text=pdf_text,
                        query=query
                    )

                    new_pdf_list.append({
                        "id": message['id'],
                        "subject": subject,
                        "attachment": filename,
                        "gpt_summary": gpt_response
                    })

        request.session['search_query'] = query
        request.session['pdf_list'] = new_pdf_list
        pdf_list = new_pdf_list

    # Handle local pagination
    total_items = len(pdf_list)
    page_size = 5

    if total_items == 0:
        return templates.TemplateResponse("results.html", {
            "request": request,
            "pdfs": [],
            "current_page": 1,
            "total_pages": 1,
            "next_page": None,
            "prev_page": None,
            "query": query
        })

    start_index = (page - 1) * page_size
    end_index = start_index + page_size

    # In case user manually enters a too-large page number
    if start_index >= total_items:
        page = 1
        start_index = 0
        end_index = page_size

    pdf_list_page = pdf_list[start_index:end_index]
    total_pages = (total_items + page_size - 1) // page_size

    next_page = page + 1 if page < total_pages else None
    prev_page = page - 1 if page > 1 else None

    print(
        f"📄 Displaying page {page} of {total_pages} "
        f"({len(pdf_list_page)} items on this page) "
        f"out of total {total_items}"
    )

    return templates.TemplateResponse("results.html", {
        "request": request,
        "pdfs": pdf_list_page,
        "current_page": page,
        "total_pages": total_pages,
        "next_page": next_page,
        "prev_page": prev_page,
        "query": query
    })


@app.get("/view_pdf", response_class=HTMLResponse)
async def view_pdf(request: Request, message_id: str):
    """View a specific PDF attachment in the browser."""
    print(f"🔍 Received request for message_id: {message_id}")

    user = request.session.get('user')
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    token = request.session.get('token')
    credentials = Credentials(
        token=token['access_token'],
        refresh_token=token.get('refresh_token'),
        token_uri='https://oauth2.googleapis.com/token',
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET
    )
    gmail_service = build('gmail', 'v1', credentials=credentials)

    try:
        msg = gmail_service.users().messages().get(userId='me', id=message_id).execute()
    except Exception as e:
        print(f"❌ Error retrieving message: {e}")
        raise HTTPException(status_code=404, detail="Email message not found")

    attachments = msg['payload'].get('parts', [])
    for attachment in attachments:
        filename = attachment.get('filename', '')
        if filename.endswith('.pdf'):
            attachment_id = attachment['body'].get('attachmentId')
            if not attachment_id:
                continue

            # Instead of re-decoding each time, we could also store
            # the raw PDF in the cache. But for demonstration:
            attachment_data = gmail_service.users().messages().attachments().get(
                userId='me', messageId=message_id, id=attachment_id
            ).execute()

            pdf_data = base64.urlsafe_b64decode(attachment_data['data'].encode('UTF-8'))
            return HTMLResponse(content=f"""
                <html>
                <body>
                    <embed src="data:application/pdf;base64,{base64.b64encode(pdf_data).decode()}"
                           width="100%" height="800px" />
                </body>
                </html>
            """, status_code=200)

    return HTMLResponse("<h1>PDF not found</h1>", status_code=404)

# ----------------------------------------------------------------------------
# 5. CACHED HELPER FUNCTIONS
# ----------------------------------------------------------------------------

def get_cached_pdf_text(
    gmail_service,
    message_id: str,
    filename: str,
    attachment_id: str
) -> str:
    """
    Returns the extracted PDF text (first 100 sentences) from cache if available,
    otherwise fetches from Gmail, extracts, and saves in `pdf_text_cache`.
    """
    cache_key = (message_id, filename)
    if cache_key in pdf_text_cache:
        return pdf_text_cache[cache_key]

    # Otherwise, fetch & decode once
    attachment_data = gmail_service.users().messages().attachments().get(
        userId='me', messageId=message_id, id=attachment_id
    ).execute()
    data = attachment_data['data']
    pdf_data = base64.urlsafe_b64decode(data.encode('UTF-8'))

    pdf_text = extract_pdf_first_100_sentences(pdf_data)
    # Store in cache
    pdf_text_cache[cache_key] = pdf_text
    return pdf_text

def get_cached_gpt_summary(
    message_id: str,
    filename: str,
    pdf_text: str,
    query: str
) -> str:
    """
    Returns GPT summary from cache if available, otherwise calls GPT
    and stores the result in `summary_cache`.
    """
    cache_key = (message_id, filename, query)
    if cache_key in summary_cache:
        return summary_cache[cache_key]

    # Otherwise, generate summary
    gpt_response = analyze_pdf_with_gpt(pdf_text, query)
    summary_cache[cache_key] = gpt_response
    return gpt_response

# ----------------------------------------------------------------------------
# 6. EXTRACTION & GPT
# ----------------------------------------------------------------------------

def extract_pdf_first_100_sentences(pdf_data: bytes) -> str:
    """Extracts the first 100 sentences from a PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_data))
        all_text = []
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                all_text.append(page_text)

        text = " ".join(all_text)
        # For speed, you might split on linebreaks instead of periods.
        # Also, you might reduce from 100 to 50 or 20 if you want to speed it up further.
        sentences = text.split('.')
        first_100 = '. '.join(sentences[:100])
        return first_100.strip()
    except Exception as e:
        return f"Error processing PDF: {e}"

def analyze_pdf_with_gpt(pdf_content: str, query: str) -> str:
    """
    Uses OpenAI GPT to analyze PDF content and provide a summary
    based on the user's 'query'.
    """
    try:
        prompt = (
            f"Analyze the content below to classify it according to the query: '{query}'\n\n"
            f"Content:\n{pdf_content}\n"
            f"Provide a concise summary."
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional assistant with expertise in text analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error analyzing PDF content: {e}"

# ----------------------------------------------------------------------------
# 7. MAIN ENTRY POINT
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
