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


@app.get("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    query: str = Query(..., description="Search query"),
    page: int = Query(1, ge=1, description="Page number"),
):
    """
    Searches PDF attachments in Gmail, uses GPT to summarize them, and
    implements local pagination (5 items per page).
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

        # If you want to incorporate the user's query into the Gmail search, use:
        #   search_query = f"has:attachment filename:pdf {query}"
        # or keep it only for GPT analysis if you prefer:
        search_query = "has:attachment filename:pdf"

        # Retrieve *all* messages by handling nextPageToken
        all_messages = []
        response = gmail_service.users().messages().list(userId='me', q=search_query).execute()
        while True:
            # messages might be None if no matches
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

            # The 'payload' could be structured in different ways
            # We check 'parts' for attachments
            parts = msg['payload'].get('parts', [])
            for part in parts:
                filename = part.get('filename', '')
                if filename.endswith('.pdf'):
                    body = part.get('body', {})
                    attachment_id = body.get('attachmentId')
                    if not attachment_id:
                        continue  # skip non-attachment parts

                    attachment_data = gmail_service.users().messages().attachments().get(
                        userId='me', messageId=message['id'], id=attachment_id
                    ).execute()

                    data = attachment_data['data']
                    pdf_data = base64.urlsafe_b64decode(data.encode('UTF-8'))

                    pdf_text_chunks = extract_pdf_first_100_sentences(pdf_data)
                    gpt_response = analyze_pdf_with_gpt(pdf_text_chunks, query)

                    new_pdf_list.append({
                        "id": message['id'],
                        "subject": subject,
                        "attachment": filename,
                        "gpt_summary": gpt_response
                    })

        # Store in session
        request.session['search_query'] = query
        request.session['pdf_list'] = new_pdf_list
        pdf_list = new_pdf_list

    # Handle local pagination
    total_items = len(pdf_list)
    page_size = 5

    # If no PDFs at all:
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

    # Calculate slicing
    start_index = (page - 1) * page_size
    end_index = start_index + page_size

    # In case user manually enters page bigger than total pages
    if start_index >= total_items:
        page = 1
        start_index = 0
        end_index = page_size

    pdf_list_page = pdf_list[start_index:end_index]

    # Determine total pages (ceiling of total_items/page_size)
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
        print(f"📩 Fetched email: {msg}")
    except Exception as e:
        print(f"❌ Error retrieving message: {e}")
        raise HTTPException(status_code=404, detail="Email message not found")

    attachments = msg['payload'].get('parts', [])
    print(f"📎 Attachments found: {attachments}")

    for attachment in attachments:
        filename = attachment.get('filename', '')
        print(f"🔎 Checking attachment: {filename}")
        if filename.endswith('.pdf'):
            attachment_id = attachment['body'].get('attachmentId')
            if not attachment_id:
                continue
            attachment_data = gmail_service.users().messages().attachments().get(
                userId='me', messageId=message_id, id=attachment_id
            ).execute()

            pdf_data = base64.urlsafe_b64decode(attachment_data['data'].encode('UTF-8'))
            print("✅ PDF found and decoded!")

            # Return inline PDF in the browser
            return HTMLResponse(content=f"""
                <html>
                <body>
                    <embed src="data:application/pdf;base64,{base64.b64encode(pdf_data).decode()}"
                           width="100%" height="800px" />
                </body>
                </html>
            """, status_code=200)

    print("❌ No PDF attachments found!")
    return HTMLResponse("<h1>PDF not found</h1>", status_code=404)


def extract_pdf_first_100_sentences(pdf_data: bytes) -> str:
    """ Extracts the first 100 sentences from a PDF file. """
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
