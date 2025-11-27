import sys
import asyncio

# FORCE WINDOWS TO USE THE CORRECT LOOP FOR PLAYWRIGHT
# This fixes "NotImplementedError" on Windows
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
# Your AI Pipe Token
AI_PIPE_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjMwMDE4MzhAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.Z8q3Ks-1adsY9WSlg6rO1pnJOUTlItzGti69W-QETTI"
AI_PIPE_BASE_URL = "https://aipipe.org/openrouter/v1"
AI_PIPE_MODEL = "openai/gpt-4o-mini" 

# Student Details
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

# --- TOOLS: FETCHERS & PARSERS ---

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
            content_str = df.to_csv(index=False)
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
            # Try decoding as plain text (JSON, TXT, HTML)
            content_str = content_bytes.decode("utf-8", errors="ignore")
            
        return content_str[:20000] # Limit size
        
    except Exception as e:
        print(f"File processing failed: {e}")
        return f"Error reading file: {str(e)}"

async def fetch_page_content(url: str):
    """
    Smart fetcher: Handles HTML pages (via Playwright) AND direct files/APIs (via HTTPX).
    """
    print(f"Fetching content from: {url}")
    
    # 1. Try direct HTTP get first (faster for APIs/Files)
    try:
        # Check if it looks like a file or API
        if any(ext in url.lower() for ext in ['.csv', '.pdf', '.json', '.xlsx', '/api/']):
            return await download_and_parse_file(url)
    except:
        pass 

    # 2. Fallback to Playwright (for JS-heavy HTML)
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
            page = await context.new_page()
            
            await page.goto(url, timeout=30000)
            
            # CRITICAL: Wait for JS to load content (Fixes Vercel/SPA issues)
            await page.wait_for_timeout(2000) 
            await page.wait_for_load_state("networkidle")
            
            # CRITICAL: Use innerHTML to see hidden elements (Fixes Hidden Password task)
            content = await page.evaluate("document.body.innerHTML") 
            await browser.close()
            return content
    except Exception as e:
        return f"Error fetching {url}: {e}"

# --- MAIN SOLVER ---
@app.post("/run")
async def solve_quiz(request: QuizRequest):
    print(f"\n=== NEW TASK: {request.url} ===")
    
    if request.secret != STUDENT_SECRET:
        raise HTTPException(status_code=403, detail="Invalid Secret")

    # "Conversation History" to keep track of what we've seen
    data_context = f"Initial Request URL: {request.url}\n"
    
    # Initial Fetch
    initial_content = await fetch_page_content(request.url)
    data_context += f"\n--- CONTENT OF {request.url} ---\n{initial_content[:15000]}\n"

    # THE LOOP: Iterate up to 5 times to handle pagination or multi-step logic
    answer = None
    submit_url = None
    
    for step in range(5):
        print(f"Step {step + 1}: Analyzing & Reasoning...")
        
        prompt = f"""
        You are an autonomous agent solving a quiz.
        
        CURRENT DATA CONTEXT:
        {data_context}
        
        YOUR GOAL:
        1. Read the content. Understand the Question.
        2. If the answer is NOT in the current data, determine which URL to fetch next (e.g., next page in pagination, or a file link).
        3. Look for the 'submit_url' in the text/code.
        4. If you found the answer, output it.
        
        OUTPUT JSON ONLY:
        {{
            "reasoning": "Explanation of what to do next...",
            "fetch_new_url": "https://... (only if you need more data, else null)",
            "answer": "The final answer (or null if fetching more data)",
            "submit_url": "https://... (extracted from text, keep previous if not found)"
        }}
        """
        
        try:
            response = await client.chat.completions.create(
                model=AI_PIPE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            decision = json.loads(response.choices[0].message.content)
            print(f"AI Decision: {decision.get('reasoning')}")
            
            # Update submit_url if found
            if decision.get("submit_url"):
                submit_url = decision["submit_url"]

            # CHECK: Do we have the answer?
            if decision.get("answer") is not None:
                answer = decision["answer"]
                break # EXIT LOOP -> We are done!
            
            # CHECK: Do we need to fetch more?
            new_url = decision.get("fetch_new_url")
            if new_url:
                print(f"Agent requested new URL: {new_url}")
                new_content = await fetch_page_content(new_url)
                # Append new data to context so LLM sees everything
                data_context += f"\n--- CONTENT OF {new_url} ---\n{new_content[:10000]}\n"
            else:
                print("AI is stuck (no answer, no new URL). Stopping.")
                break
                
        except Exception as e:
            print(f"Loop Error: {e}")
            break

    # Final Submission
    if not submit_url:
        # Fallback logic: sometimes LLM misses it, maybe it was in the first prompt? 
        # For now, we return error if missing.
        return {"status": "error", "message": "No submission URL found."}

    payload = {
        "email": STUDENT_EMAIL,
        "secret": STUDENT_SECRET,
        "url": request.url,
        "answer": answer
    }
    
    print(f"Submitting answer: {answer} to {submit_url}")
    try:
        async with httpx.AsyncClient() as post_client:
            sub_resp = await post_client.post(submit_url, json=payload, timeout=10)
            
            # Return strict format expected by the server
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
