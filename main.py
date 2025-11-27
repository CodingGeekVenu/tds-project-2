import sys
import asyncio

# FORCE WINDOWS TO USE THE CORRECT LOOP FOR PLAYWRIGHT
# This fixes the "NotImplementedError" on Windows machines
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import os
import json
import httpx
import pandas as pd
import io
from pypdf import PdfReader
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from playwright.async_api import async_playwright
from openai import AsyncOpenAI
from contextlib import asynccontextmanager

# --- CONFIGURATION ---
# Your specific AI Pipe Token
AI_PIPE_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjMwMDE4MzhAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.Z8q3Ks-1adsY9WSlg6rO1pnJOUTlItzGti69W-QETTI"
AI_PIPE_BASE_URL = "https://aipipe.org/openrouter/v1"
AI_PIPE_MODEL = "openai/gpt-4o-mini" 

# Your Student Details
STUDENT_EMAIL = "23f3001838@ds.study.iitm.ac.in"
STUDENT_SECRET = "TDS_2025_GenAI"

# Initialize AI Client
client = AsyncOpenAI(api_key=AI_PIPE_TOKEN, base_url=AI_PIPE_BASE_URL)

# --- KEEP ALIVE (For Render Free Tier) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Launch the background pinger
    task = asyncio.create_task(keep_alive_loop())
    yield
    # Shutdown
    task.cancel()

async def keep_alive_loop():
    """Pings the server itself every 10 minutes to prevent sleep."""
    while True:
        try:
            # Use the Render URL if deployed, else localhost
            my_url = os.getenv("RENDER_EXTERNAL_URL", "http://localhost:8000")
            if "localhost" not in my_url: 
                async with httpx.AsyncClient() as c:
                    await c.get(my_url)
        except: 
            pass
        await asyncio.sleep(600) # Sleep 10 minutes

app = FastAPI(lifespan=lifespan)

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str

# --- HELPER: FILE DOWNLOADER & PARSER ---
async def download_and_parse_file(file_url: str):
    """
    Downloads a file (CSV/JSON/PDF) and returns its text content.
    """
    print(f"Downloading file from: {file_url}")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(file_url, follow_redirects=True, timeout=15)
            resp.raise_for_status()
            content_bytes = resp.content
            
        # Determine file type from URL or Headers
        content_str = ""
        filename = file_url.split("/")[-1].lower()
        
        if ".csv" in filename:
            df = pd.read_csv(io.BytesIO(content_bytes))
            content_str = df.to_csv(index=False) # Convert back to string for LLM
        elif ".xlsx" in filename:
            df = pd.read_excel(io.BytesIO(content_bytes))
            content_str = df.to_csv(index=False)
        elif ".pdf" in filename:
            reader = PdfReader(io.BytesIO(content_bytes))
            text = []
            for page in reader.pages:
                text.append(page.extract_text())
            content_str = "\n".join(text)
        else:
            # Try decoding as plain text
            content_str = content_bytes.decode("utf-8", errors="ignore")
            
        # Limit content size for LLM (GPT-4o-mini limit)
        return content_str[:20000] 
        
    except Exception as e:
        print(f"File processing failed: {e}")
        return f"Error reading file: {str(e)}"

# --- MAIN ENDPOINT ---
@app.post("/run")
async def solve_quiz(request: QuizRequest):
    print(f"\n=== NEW TASK: {request.url} ===")
    
    # Security Check
    if request.secret != STUDENT_SECRET:
        raise HTTPException(status_code=403, detail="Invalid Secret")

    # 1. Scrape the Main Page
    scraped_text = ""
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
            page = await context.new_page()
            
            # Go to the page
            await page.goto(request.url, timeout=30000)
            
            # --- CRITICAL FIX FOR DYNAMIC CONTENT & HIDDEN ELEMENTS ---
            # 1. Wait 2 seconds for any JavaScript animations/loading to finish
            await page.wait_for_timeout(2000)
            await page.wait_for_load_state("networkidle")
            
            # 2. Use innerHTML (not innerText) so the LLM can see hidden divs,
            # inputs, and raw structure (needed for the Password task).
            scraped_text = await page.evaluate("document.body.innerHTML")
            
            # Also get all links (hrefs) in case the LLM needs to find a file link
            links = await page.evaluate("""
                Array.from(document.querySelectorAll('a')).map(a => ({text: a.innerText, href: a.href}))
            """)
            
            await browser.close()
    except Exception as e:
        print(f"Scraping Failed: {e}")
        raise HTTPException(status_code=500, detail="Scraping failed")

    # 2. Phase 1: Analyze & Plan
    print("Phase 1 Analysis: Checking for files...")
    links_json = json.dumps(links[:20]) # Limit to first 20 links to save tokens
    
    # We tell the LLM to look at the HTML structure
    prompt_phase1 = f"""
    You are a data analysis agent.
    
    PAGE HTML CONTENT:
    {scraped_text[:15000]} 
    
    LINKS ON PAGE:
    {links_json}
    
    TASK:
    1. Read the content. Look for questions, hidden keys, or data tasks.
    2. Does the question require downloading a file (CSV, PDF, etc)?
    3. If YES, return the 'file_url'. If NO, return null.
    4. Extract the 'submit_url' (look for form actions or text saying "submit to").
    
    OUTPUT JSON ONLY:
    {{
        "requires_file": true/false,
        "file_url": "https://...",
        "submit_url": "https://..."
    }}
    """
    
    try:
        resp1 = await client.chat.completions.create(
            model=AI_PIPE_MODEL,
            messages=[{"role": "user", "content": prompt_phase1}],
            response_format={"type": "json_object"}
        )
        plan = json.loads(resp1.choices[0].message.content)
        print(f"Plan: {plan}")
    except Exception as e:
        return {"error": "Planning failed", "details": str(e)}

    # 3. Execute File Download (If needed)
    additional_context = ""
    if plan.get("requires_file") and plan.get("file_url"):
        file_url = plan["file_url"]
        # Handle relative URLs if necessary (assuming absolute for now or LLM fixes it)
        file_content = await download_and_parse_file(file_url)
        additional_context = f"\n\n--- DOWNLOADED FILE CONTENT ---\n{file_content}\n--- END FILE ---\n"

    # 4. Phase 2: Final Solve
    print("Phase 2: Solving...")
    final_prompt = f"""
    You are a quiz solver.
    
    ORIGINAL PAGE HTML:
    {scraped_text[:15000]}
    
    {additional_context}
    
    TASK:
    1. Solve the question found in the HTML. 
    2. If there is a hidden key or reversed text, decode it.
    3. If you have file content, use it to calculate the answer.
    4. Return the answer in JSON format.
    
    OUTPUT JSON ONLY:
    {{
        "answer": <value>
    }}
    """
    
    try:
        resp2 = await client.chat.completions.create(
            model=AI_PIPE_MODEL,
            messages=[{"role": "user", "content": final_prompt}],
            response_format={"type": "json_object"}
        )
        final_result = json.loads(resp2.choices[0].message.content)
        answer = final_result.get("answer")
        print(f"Calculated Answer: {answer}")
    except Exception as e:
        return {"error": "Solving failed", "details": str(e)}

    # 5. Submit the Answer
    submit_url = plan.get("submit_url")
    if not submit_url:
        return {"error": "No submission URL found"}

    payload = {
        "email": STUDENT_EMAIL,
        "secret": STUDENT_SECRET,
        "url": request.url,
        "answer": answer
    }
    
    print(f"Submitting to {submit_url}...")
    try:
        async with httpx.AsyncClient() as post_client:
            sub_resp = await post_client.post(submit_url, json=payload, timeout=10)
            
            # Return the specific format the server expects (or just a success message)
            return {
                "status": "success", 
                "submitted_to": submit_url,
                "server_response": sub_resp.json() if sub_resp.status_code == 200 else sub_resp.text
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
