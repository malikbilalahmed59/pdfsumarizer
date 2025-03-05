from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from authlib.integrations.starlette_client import OAuth, OAuthError
from starlette.middleware.sessions import SessionMiddleware
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaInMemoryUpload
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
import os
import base64
import openai
import PyPDF2
import io
from google.auth.exceptions import RefreshError

# ----------------------------------------------------------------------------
# 1) Environment + Global Caches
# ----------------------------------------------------------------------------

load_dotenv()
CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')
OPENAI_APIKEY = os.getenv('OPENAI_APIKEY')
SECRET_KEY = os.getenv('SECRET_KEY')

# Simple in-memory caches (demo; for production, use Redis or DB)
pdf_text_cache = {}    # key: (message_id, filename) -> extracted text
summary_cache = {}     # key: (message_id, filename, query) -> GPT summary

# ----------------------------------------------------------------------------
# 2) Create FastAPI + Session Middleware
# ----------------------------------------------------------------------------

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

templates = Jinja2Templates(directory="templates")  # assume your HTML files live in ./static
app.mount("/static", StaticFiles(directory="static"), name="static")

# ----------------------------------------------------------------------------
# 3) Configure OAuth (Gmail & Drive scopes)
# ----------------------------------------------------------------------------

oauth = OAuth()
oauth.register(
    name='google',
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    # This URL points to Google's OIDC configuration
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={
        # We’re requesting both Gmail read scope and Drive file scope
        'scope': (
            'openid email profile '
            'https://www.googleapis.com/auth/gmail.readonly '
            'https://www.googleapis.com/auth/drive.file'
        ),
        'access_type': 'offline',
        'prompt': 'consent'
    }
)

openai.api_key = OPENAI_APIKEY

# ----------------------------------------------------------------------------
# 4) Routes: Login, Logout, Home
# ----------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """ Home Page: Redirects user to login if not authenticated """
    user = request.session.get('user')

    return templates.TemplateResponse("home.html", {"request": request, "user": user})


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
    """ Logs out the user and clears session """
    request.session.pop('user', None)
    request.session.pop('token', None)
    request.session.pop('search_query', None)
    request.session.pop('pdf_list', None)
    return RedirectResponse(url='/')

# ----------------------------------------------------------------------------
# 5) Search Route (Gmail + GPT + Pagination)
# ----------------------------------------------------------------------------

@app.get("/search", response_class=HTMLResponse)
async def search(
        request: Request,
        query: str = Query(..., description="Search query"),
        page: int = Query(1, ge=1, description="Page number"),
):
    """ Searches PDF attachments in Gmail with proper pagination (fetch 5 PDFs at a time). """

    user = request.session.get('user', {})
    if not user:
        return RedirectResponse(url="/logout")  # Auto logout if user is not authenticated

    token = request.session.get('token')

    if not token:
        return RedirectResponse(url="/logout")  # Auto logout if token is missing

    try:
        # ✅ Initialize credentials
        credentials = Credentials(
            token=token['access_token'],
            refresh_token=token.get('refresh_token'),
            token_uri='https://oauth2.googleapis.com/token',
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET
        )

        # ✅ Initialize Gmail API
        gmail_service = build('gmail', 'v1', credentials=credentials)
        search_query = f"in:anywhere has:attachment filename:pdf {query}"

        # ✅ Retrieve pagination tokens from session
        next_page_tokens = request.session.get('next_page_tokens', {})

        # ✅ Determine which page token to use
        page_token = next_page_tokens.get(page, None)

        # ✅ Fetch only the necessary messages (current batch)
        page_size = 5  # Process only 5 PDFs per request
        response = gmail_service.users().messages().list(userId='me', q=search_query, maxResults=page_size,
                                                         pageToken=page_token).execute()
        messages = response.get('messages', [])

        # ✅ Store the nextPageToken for future pages
        if 'nextPageToken' in response:
            next_page_tokens[page + 1] = response['nextPageToken']
            request.session['next_page_tokens'] = next_page_tokens  # Save tokens in session

        print(f"📬 Fetching {len(messages)} PDFs for page {page}")

        pdf_list = []
        for message in messages:
            msg = gmail_service.users().messages().get(userId='me', id=message['id']).execute()
            subject = next((header['value'] for header in msg['payload']['headers'] if header['name'] == 'Subject'),
                           "No Subject")

            # Process each attachment
            for part in msg['payload'].get('parts', []):
                filename = part.get('filename', '')
                if filename.endswith('.pdf'):
                    attachment_id = part['body'].get('attachmentId')
                    if not attachment_id:
                        continue

                    # ✅ Process only 5 PDFs at a time
                    pdf_text = get_cached_pdf_text(gmail_service, message['id'], filename, attachment_id)
                    gpt_response = get_cached_gpt_summary(message['id'], filename, pdf_text, query)

                    pdf_list.append({
                        "id": message['id'],
                        "subject": subject,
                        "attachment": filename,
                        "gpt_summary": gpt_response
                    })

        # ✅ Check if there's a next page (if nextPageToken exists)
        next_page = page + 1 if 'nextPageToken' in response else None
        prev_page = page - 1 if page > 1 else None

        print(f"📄 Displaying {len(pdf_list)} PDFs for page {page}, next page: {next_page}")

        return templates.TemplateResponse("results.html", {
            "request": request,
            "user": user,
            "pdfs": pdf_list,
            "current_page": page,
            "next_page": next_page,
            "prev_page": prev_page,
            "query": query
        })

    except RefreshError:
        print("🔴 Token expired. Logging out user...")
        request.session.clear()  # Clear the session
        return RedirectResponse(url="/logout")
# ----------------------------------------------------------------------------
# 6) View PDF Route
# ----------------------------------------------------------------------------

@app.get("/view_pdf", response_class=HTMLResponse)
async def view_pdf(request: Request, message_id: str):
    """
    Fetches the PDF from Gmail and returns it inline in the browser.
    """
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
# 7) Save to Drive Route
# ----------------------------------------------------------------------------

@app.get("/save_to_drive", response_class=HTMLResponse)
async def save_to_drive(request: Request, message_id: str, filename: str):
    """
    Fetches the PDF from Gmail and uploads it to a folder named 'AI PDF Finder' in the user's Drive.
    """
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
    drive_service = build('drive', 'v3', credentials=credentials)

    # 1) Fetch PDF from Gmail
    try:
        msg = gmail_service.users().messages().get(userId='me', id=message_id).execute()
        parts = msg['payload'].get('parts', [])
        pdf_data = None

        for part in parts:
            part_filename = part.get('filename', '')
            if part_filename == filename:
                # Found the correct PDF part
                attachment_id = part['body'].get('attachmentId')
                attachment_res = gmail_service.users().messages().attachments().get(
                    userId='me', messageId=message_id, id=attachment_id
                ).execute()
                pdf_data = base64.urlsafe_b64decode(attachment_res['data'].encode('UTF-8'))
                break

        if pdf_data is None:
            return HTMLResponse("<h1>PDF attachment not found.</h1>", status_code=404)
    except HttpError as e:
        return HTMLResponse(f"<h1>Error fetching email: {e}</h1>", status_code=400)

    # 2) Find or create the "AI PDF Finder" folder
    try:
        folder_id = find_or_create_folder(drive_service, "AI PDF Finder")
    except HttpError as e:
        return HTMLResponse(f"<h1>Error creating/finding folder: {e}</h1>", status_code=400)

    # 3) Upload the PDF
    try:
        uploaded_file = upload_pdf_to_drive(drive_service, folder_id, pdf_data, filename)
    except HttpError as e:
        return HTMLResponse(f"<h1>Error uploading file to Drive: {e}</h1>", status_code=400)

    # 4) Confirm success
    return HTMLResponse(f"""
    <html>
    <body>
        <h2>Success!</h2>
        <p>Uploaded <strong>{filename}</strong> to your Google Drive folder: AI PDF Finder</p>
        <p>View in Drive:
            <a href="https://drive.google.com/drive/folders/{folder_id}" target="_blank">AI PDF Finder</a>
        </p>
        <a href="/">Return Home</a>
    </body>
    </html>
    """, status_code=200)


def find_or_create_folder(drive_service, folder_name: str) -> str:
    """
    Searches for a folder in Drive by name. If it doesn't exist, create it.
    Returns the folder ID.
    """
    query = f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    results = drive_service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get('files', [])
    if files:
        return files[0]['id']  # folder already exists

    # Otherwise, create it
    file_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    folder = drive_service.files().create(body=file_metadata, fields='id').execute()
    return folder['id']


def upload_pdf_to_drive(drive_service, folder_id: str, pdf_data: bytes, filename: str):
    """
    Uploads the given PDF content to the specified Drive folder.
    Returns the uploaded file metadata.
    """
    file_metadata = {
        'name': filename,
        'parents': [folder_id]
    }
    media = MediaInMemoryUpload(pdf_data, mimetype='application/pdf')
    uploaded_file = drive_service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id, name'
    ).execute()
    return uploaded_file

# ----------------------------------------------------------------------------
# 8) Cache Helpers + GPT
# ----------------------------------------------------------------------------

def get_cached_pdf_text(
    gmail_service,
    message_id: str,
    filename: str,
    attachment_id: str
) -> str:
    """
    Returns extracted PDF text (first 100 sentences) from cache if available,
    otherwise fetches & extracts from Gmail and stores in pdf_text_cache.
    """
    cache_key = (message_id, filename)
    if cache_key in pdf_text_cache:
        return pdf_text_cache[cache_key]

    attachment_data = gmail_service.users().messages().attachments().get(
        userId='me', messageId=message_id, id=attachment_id
    ).execute()
    data = attachment_data['data']
    pdf_data = base64.urlsafe_b64decode(data.encode('UTF-8'))

    pdf_text = extract_pdf_first_100_sentences(pdf_data)
    pdf_text_cache[cache_key] = pdf_text
    return pdf_text


def get_cached_gpt_summary(message_id: str, filename: str, pdf_text: str, query: str) -> str:
    """
    Returns GPT summary from cache if available, otherwise calls GPT
    and caches the result.
    """
    cache_key = (message_id, filename, query)
    if cache_key in summary_cache:
        return summary_cache[cache_key]

    gpt_response = analyze_pdf_with_gpt(pdf_text, query)
    summary_cache[cache_key] = gpt_response
    return gpt_response


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
# 9) Main Entry
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
