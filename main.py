from agents import Agent, Runner, OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json
import asyncio
import traceback

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL")

# Validation
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment")

client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url=GEMINI_BASE_URL
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)

full_stack_developer_agent = Agent(
    name="Full Stack Developer Agent",
    instructions="""
        Role: Expert Full Stack Developer specializing in Frontend, Backend, and FastAPI.

        Core Skills:
        - Frontend: React, Vue, Angular, JavaScript, TypeScript, CSS
        - Backend: FastAPI (specialist), Python, Node.js
        - Databases: SQL (PostgreSQL), NoSQL (MongoDB)
        - Tools: Git, Docker, REST APIs

        Response Rules:
        - Ask clarifying questions if request is unclear
        - Provide structured, code-commented solutions
        - Explain reasoning behind answers
        - Focus on security and best practices
        - Use code blocks with language tags

        Specialty: FastAPI development, frontend-backend integration, problem-solving.
        Goal: Help users build better software through clear, actionable guidance.
    """,
    model=model
)

app = FastAPI(title="Full Stack Developer Agent API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production mein specific domain use karo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

@app.get("/")
def read_root():
    return {
        "message": "Hello from Saud Ali",
        "status": "online",
        "endpoints": ["/chat", "/chat/stream", "/health"]
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "api_key_set": bool(GEMINI_API_KEY)}

class ChatRequest(BaseModel):
    message: str

    class Config:
        json_schema_extra = {
            "example": {
                "message": "How to create a REST API with FastAPI?"
            }
        }

# Non-streaming endpoint (for testing)
@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        print(f"📨 Received chat request: {req.message[:50]}...")
        
        result = await Runner.run(
            full_stack_developer_agent,
            req.message
        )
        
        print(f"✅ Response generated: {len(result.final_output)} chars")
        return {"response": result.final_output}
    
    except Exception as e:
        print(f"❌ Error in /chat: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Streaming endpoint - OPTIMIZED VERSION
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    async def event_generator():
        try:
            print(f"🔵 Stream started for: {req.message[:50]}...")
            
            # Test if message is empty
            if not req.message or not req.message.strip():
                yield f"data: {json.dumps({'error': 'Empty message'})}\n\n"
                return
            
            # Method 1: Try streaming with .stream()
            try:
                stream_result = Runner.run_streamed(
                    full_stack_developer_agent,
                    req.message
                )
                
                print(f"📡 Stream result type: {type(stream_result)}")
                
                if hasattr(stream_result, 'stream'):
                    print("✅ Using .stream() method")
                    async for chunk in stream_result.stream():
                        content = None
                        
                        if hasattr(chunk, 'delta') and chunk.delta:
                            content = chunk.delta
                        elif hasattr(chunk, 'content') and chunk.content:
                            content = chunk.content
                        elif isinstance(chunk, str):
                            content = chunk
                        
                        if content:
                            data = json.dumps({"content": content})
                            yield f"data: {data}\n\n"
                            await asyncio.sleep(0.01)
                    
                    yield f"data: {json.dumps({'done': True})}\n\n"
                    print("✅ Stream completed via .stream()")
                    return
                
            except Exception as stream_error:
                print(f"⚠️ Streaming method failed: {stream_error}")
            
            # Method 2: Fallback - Get full response and simulate streaming
            print("🔄 Falling back to simulated streaming")
            result = await Runner.run(
                full_stack_developer_agent,
                req.message
            )
            
            full_response = result.final_output
            print(f"✅ Got full response: {len(full_response)} chars")
            
            # Stream character by character (or word by word)
            chunk_size = 5  # characters per chunk
            for i in range(0, len(full_response), chunk_size):
                chunk = full_response[i:i + chunk_size]
                data = json.dumps({"content": chunk})
                yield f"data: {data}\n\n"
                await asyncio.sleep(0.02)  # Adjust speed
            
            yield f"data: {json.dumps({'done': True})}\n\n"
            print("✅ Simulated stream completed")
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            yield f"data: {json.dumps({'error': error_msg})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
