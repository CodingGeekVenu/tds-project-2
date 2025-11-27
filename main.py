import sys
import asyncio

# 1. WINDOWS FIX: Forces the correct event loop for Playwright on Windows
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
        await asyncio.sleep(600) 

app = FastAPI(lifespan=lifespan)

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str

# --- TOOLS: PARSERS & FETCHERS ---

async def download_and_parse_file(content_bytes: bytes, url: str):
    """Helper to parse binary content (PDF, CSV, Excel) into text for the LLM."""
    try:
        filename = url.split("/")[-1].lower()
        # Handle CSV
        if ".csv" in filename or "csv" in url:
            df = pd.read_csv(io.BytesIO(content_bytes))
            return df.to_csv(index=False)
        # Handle Excel
        elif ".xlsx" in filename or "excel" in url:
            df = pd.read_excel(io.BytesIO(content_bytes))
            return df.to_csv(index=False)
        # Handle PDF
        elif ".pdf" in filename or "pdf" in url:
            reader = PdfReader(io.BytesIO(content_bytes))
            text = []
            for page in reader.pages:
                text.append(page.extract_text())
            return "\n".join(text)
        # Handle Plain Text / JSON
        else:
            return content_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        return f"Error parsing file: {e}"

async def smart_fetch(url: str, method: str = "GET", headers: dict = None, data: dict = None):
    """
    Universal Fetcher: Handles GET/POST, Headers, JSON, Files, and JS-Heavy HTML.
    """
    print(f"📡 Fetching: {method} {url}")
    
    # 1. Try HTTPX first (Fastest for APIs, Files, Level 3 Auth, Level 11 POST)
    try:
        async with httpx.AsyncClient() as client:
            if method.upper() == "POST":
                resp = await client.post(url, json=data, headers=headers, follow_redirects=True, timeout=15)
            else:
                resp = await client.get(url, headers=headers, follow_redirects=True, timeout=15)
            
            # Check if it's a file or API response
            content_type = resp.headers.get("content-type", "").lower()
            
            # If it looks like data (JSON, CSV, PDF, Excel)
            if any(x in content_type for x in ["json", "csv", "pdf", "octet-stream"]) or any(x in url for x in [".csv", ".pdf", ".xlsx"]):
                if "json" in content_type:
                    return json.dumps(resp.json(), indent=2)
                return await download_and_parse_file(resp.content, url)
                
    except Exception as e:
        print(f"HTTPX fetch failed (might be JS page), trying Playwright... Error: {e}")

    # 2. Fallback to Playwright (Only supports GET, handles JS rendering & innerHTML)
    # Required for Level 1 (Hidden Elements) and Level 6 (JS Execution)
    if method.upper() == "GET":
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
                page = await context.new_page()
                
                # Set headers if needed (Level 3 Auth)
                if headers:
                    await page.set_extra_http_headers(headers)
                
                await page.goto(url, timeout=30000)
                
                # CRITICAL: Wait for JS to load content
                await page.wait_for_timeout(2000) 
                await page.wait_for_load_state("networkidle")
                
                # CRITICAL: Use innerHTML to see hidden/reversed elements
                content = await page.evaluate("document.body.innerHTML")
                await browser.close()
                return content
        except Exception as e:
            return f"Error fetching {url}: {e}"
            
    return "Error: Could not fetch content via HTTPX or Playwright."

# --- MAIN SOLVER ---
@app.post("/run")
async def solve_quiz(request: QuizRequest):
    print(f"\n=== NEW TASK: {request.url} ===")
    
    if request.secret != STUDENT_SECRET:
        raise HTTPException(status_code=403, detail="Invalid Secret")

    # "Conversation History" context
    data_context = f"Initial Request URL: {request.url}\n"
    
    # Initial Fetch
    initial_content = await smart_fetch(request.url)
    data_context += f"\n--- CONTENT OF {request.url} ---\n{initial_content[:15000]}\n"

    # THE LOOP: Up to 15 steps to handle Pagination (L2) or Pipelines (L10)
    answer = None
    submit_url = None
    
    for step in range(15):
        print(f"🧠 Step {step + 1}: Analyzing & Reasoning...")
        
        prompt = f"""
        You are an autonomous data agent capable of traversing APIs, downloading files, and solving puzzles.
        
        CURRENT DATA CONTEXT:
        {data_context}
        
        YOUR GOAL:
        1. Understand the Task.
        2. If the answer is found, output it.
        3. If you need more data (e.g. next page, new API endpoint, file download), request it.
        4. Identify the 'submit_url'.
        
        CAPABILITIES:
        - You can GET pages.
        - You can GET with Headers (e.g. for Auth).
        - You can POST JSON data (needed for some complex logic levels).
        
        OUTPUT JSON ONLY:
        {{
            "reasoning": "Explanation...",
            "tool_use": {{
                "action": "fetch", 
                "url": "https://...",
                "method": "GET", (or POST)
                "headers": {{ "Key": "Value" }}, (Optional, for Level 3)
                "data": {{ "key": "value" }} (Optional, for POST requests)
            }} (OR null if no tool needed),
            "answer": "Final answer string/number" (OR null if not ready),
            "submit_url": "https://..." (Keep looking if not found)
        }}
        """
        
        try:
            response = await client.chat.completions.create(
                model=AI_PIPE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            decision = json.loads(response.choices[0].message.content)
            print(f"🤖 Decision: {decision.get('reasoning')}")
            
            # 1. Capture Submission URL
            if decision.get("submit_url"):
                submit_url = decision["submit_url"]

            # 2. Check for Victory
            if decision.get("answer") is not None:
                answer = decision["answer"]
                break 
            
            # 3. Execute Tool (Fetch/Post)
            tool = decision.get("tool_use")
            if tool and tool.get("url"):
                method = tool.get("method", "GET")
                headers = tool.get("headers")
                data = tool.get("data")
                
                print(f"🛠 Tool: {method} {tool['url']}")
                new_content = await smart_fetch(tool['url'], method, headers, data)
                
                # Append result to context so LLM sees it in next loop
                data_context += f"\n--- RESULT FROM {tool['url']} ---\n{new_content[:10000]}\n"
            else:
                print("⚠️ AI stuck: No answer and no new tool action.")
                break
                
        except Exception as e:
            print(f"❌ Loop Error: {e}")
            break

    # Final Submission
    if not submit_url:
        return {"status": "error", "message": "No submission URL found."}

    payload = {
        "email": STUDENT_EMAIL,
        "secret": STUDENT_SECRET,
        "url": request.url,
        "answer": answer
    }
    
    print(f"🚀 Submitting answer: {answer} to {submit_url}")
    try:
        async with httpx.AsyncClient() as post_client:
            sub_resp = await post_client.post(submit_url, json=payload, timeout=15)
            
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
