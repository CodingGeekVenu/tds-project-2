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

# --- CONFIGURATION (SECURE) ---
AI_PIPE_TOKEN = os.getenv("AI_PIPE_TOKEN", "") 
AI_PIPE_BASE_URL = os.getenv("AI_PIPE_BASE_URL", "https://aipipe.org/openrouter/v1")
AI_PIPE_MODEL = os.getenv("AI_PIPE_MODEL", "openai/gpt-4o-mini") 

# Student Details
STUDENT_EMAIL = "23f3001838@ds.study.iitm.ac.in"
STUDENT_SECRET = "TDS_2025_GenAI"

# Initialize AI Client
client = AsyncOpenAI(api_key=AI_PIPE_TOKEN, base_url=AI_PIPE_BASE_URL)

# --- KEEP ALIVE ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(keep_alive_loop())
    yield
    task.cancel()

async def keep_alive_loop():
    while True:
        try:
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

# --- TOOLS ---

async def download_and_parse_file(content_bytes: bytes, url: str):
    try:
        filename = url.split("/")[-1].lower()
        if ".csv" in filename or "csv" in url:
            df = pd.read_csv(io.BytesIO(content_bytes))
            return df.to_csv(index=False)
        elif ".xlsx" in filename or "excel" in url:
            df = pd.read_excel(io.BytesIO(content_bytes))
            return df.to_csv(index=False)
        elif ".pdf" in filename or "pdf" in url:
            reader = PdfReader(io.BytesIO(content_bytes))
            text = []
            for page in reader.pages:
                text.append(page.extract_text())
            return "\n".join(text)
        else:
            return content_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        return f"Error parsing file: {e}"

async def smart_fetch(url: str, method: str = "GET", headers: dict = None, data: dict = None):
    print(f"📡 Fetching: {method} {url}")
    try:
        async with httpx.AsyncClient() as client:
            if method.upper() == "POST":
                resp = await client.post(url, json=data, headers=headers, follow_redirects=True, timeout=15)
            else:
                resp = await client.get(url, headers=headers, follow_redirects=True, timeout=15)
            
            content_type = resp.headers.get("content-type", "").lower()
            if any(x in content_type for x in ["json", "csv", "pdf", "octet-stream"]) or any(x in url for x in [".csv", ".pdf", ".xlsx"]):
                if "json" in content_type:
                    return json.dumps(resp.json(), indent=2)
                return await download_and_parse_file(resp.content, url)
                
    except Exception as e:
        print(f"HTTPX fetch failed, trying Playwright... Error: {e}")

    if method.upper() == "GET":
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
                page = await context.new_page()
                
                if headers:
                    await page.set_extra_http_headers(headers)
                
                await page.goto(url, timeout=30000)
                await page.wait_for_timeout(2000) 
                await page.wait_for_load_state("networkidle")
                content = await page.evaluate("document.body.innerHTML")
                await browser.close()
                return content
        except Exception as e:
            return f"Error fetching {url}: {e}"
            
    return "Error: Could not fetch content."

# --- MAIN SOLVER ---
@app.post("/run")
async def solve_quiz(request: QuizRequest):
    print(f"\n=== NEW TASK: {request.url} ===")
    
    if request.secret != STUDENT_SECRET:
        raise HTTPException(status_code=403, detail="Invalid Secret")

    data_context = f"Initial Request URL: {request.url}\n"
    initial_content = await smart_fetch(request.url)
    data_context += f"\n--- CONTENT OF {request.url} ---\n{initial_content[:15000]}\n"

    answer = None
    submit_url = None
    
    for step in range(15):
        print(f"🧠 Step {step + 1}: Analyzing...")
        
        # --- KEY UPDATE: Added Logic for Submit URL ---
        prompt = f"""
        You are an autonomous data agent.
        
        YOUR IDENTITY:
        Email: {STUDENT_EMAIL}
        Secret: {STUDENT_SECRET}
        
        CURRENT DATA CONTEXT:
        {data_context}
        
        YOUR GOAL:
        1. Understand the Question/Task.
        2. If you need more data (e.g. next page, API, file), use the 'fetch' tool.
        3. Identify the 'submit_url'.
        4. If you found the answer, output it.
        
        IMPORTANT:
        - The 'submit_url' is almost ALWAYS "https://tds-llm-analysis.s-anand.net/submit". Use this unless the page explicitly gives a different FULL URL for submission.
        - The text "url = ..." usually refers to the JSON payload, NOT the submission endpoint.
        - If the page mentions "project2" or "How to play" and asks you to start, the answer is usually an empty JSON object: {{}}
        - If the answer is found, set it in the "answer" field. Do NOT return null for answer if you are ready to submit.
        
        OUTPUT JSON ONLY:
        {{
            "reasoning": "Explanation...",
            "tool_use": {{
                "action": "fetch", 
                "url": "https://...",
                "method": "GET", (or POST)
                "headers": {{ "Key": "Value" }}, 
                "data": {{ "key": "value" }}
            }} (OR null),
            "answer": "Final answer string/number/json_string" (OR null if not ready),
            "submit_url": "https://..." 
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
            
            if decision.get("submit_url"):
                submit_url = decision["submit_url"]

            # Capture answer if provided
            if decision.get("answer") is not None:
                answer = decision["answer"]
                break 
            
            tool = decision.get("tool_use")
            if tool and tool.get("url"):
                method = tool.get("method", "GET")
                headers = tool.get("headers")
                data = tool.get("data")
                print(f"🛠 Tool: {method} {tool['url']}")
                new_content = await smart_fetch(tool['url'], method, headers, data)
                data_context += f"\n--- RESULT FROM {tool['url']} ---\n{new_content[:10000]}\n"
            else:
                # If AI decides to submit but returned 'answer: null' (common error), we fallback to "{}"
                if submit_url and answer is None:
                    print("⚠️ AI ready to submit but gave no answer. Defaulting to '{}' for start page.")
                    answer = "{}"
                    break
                
                print("⚠️ AI stuck.")
                break
                
        except Exception as e:
            print(f"❌ Loop Error: {e}")
            break

    # --- FALLBACK: Force correct submit URL if missing or wrong ---
    if not submit_url:
        submit_url = "https://tds-llm-analysis.s-anand.net/submit"
    
    # Fallback to empty JSON if answer is still None
    if answer is None:
        answer = "{}"

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
