from agents import Agent, Runner, OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json
import asyncio
import traceback

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL")

# Startup validation
print("=" * 50)
print("🚀 Starting Chatbot Backend...")
print(f"📍 API Key Present: {bool(GEMINI_API_KEY)}")
print(f"📍 API Key (first 10 chars): {GEMINI_API_KEY[:10] if GEMINI_API_KEY else 'MISSING'}")
print(f"📍 Base URL: {GEMINI_BASE_URL}")
print("=" * 50)

if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY is not set in environment variables!")

if not GEMINI_BASE_URL:
    raise ValueError("❌ GEMINI_BASE_URL is not set in environment variables!")

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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "message": "Hello from Saud Ali",
        "status": "running",
        "model": "gemini-2.0-flash",
        "api_configured": bool(GEMINI_API_KEY)
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "api_key_present": bool(GEMINI_API_KEY),
        "base_url": GEMINI_BASE_URL,
        "model": "gemini-2.0-flash"
    }

class ChatRequest(BaseModel):
    message: str

# Non-streaming endpoint - for testing
@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        print(f"\n{'='*50}")
        print(f"📥 Received message: {req.message}")
        
        result = await Runner.run(
            full_stack_developer_agent,
            req.message
        )
        
        print(f"✅ Response generated successfully")
        print(f"📤 Response length: {len(result.final_output)}")
        print(f"{'='*50}\n")
        
        return {"response": result.final_output}
    
    except Exception as e:
        print(f"❌ Error in /chat: {str(e)}")
        traceback.print_exc()
        return {"error": str(e)}

# Streaming endpoint - IMPROVED VERSION
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    async def event_generator():
        try:
            print(f"\n{'='*50}")
            print(f"📥 Stream Request: {req.message}")
            print(f"🔄 Starting streaming response...")
            
            # Try streaming first
            try:
                print("🔍 Attempting run_streamed...")
                stream_result = Runner.run_streamed(
                    full_stack_developer_agent,
                    req.message
                )
                
                print(f"📊 Stream result type: {type(stream_result)}")
                
                # Method 1: Check for .stream()
                if hasattr(stream_result, 'stream'):
                    print("✅ Using .stream() method")
                    chunk_count = 0
                    async for chunk in stream_result.stream():
                        content = None
                        
                        if hasattr(chunk, 'delta') and chunk.delta:
                            content = chunk.delta
                        elif hasattr(chunk, 'content') and chunk.content:
                            content = chunk.content
                        elif isinstance(chunk, str):
                            content = chunk
                        
                        if content:
                            chunk_count += 1
                            data = json.dumps({"content": content})
                            yield f"data: {data}\n\n"
                            print(f"📤 Chunk {chunk_count}: {content[:50]}...")
                            await asyncio.sleep(0.01)
                    
                    print(f"✅ Streamed {chunk_count} chunks")
                
                # Method 2: Direct async iteration
                elif hasattr(stream_result, '__aiter__'):
                    print("✅ Using async iteration")
                    chunk_count = 0
                    async for chunk in stream_result:
                        content = None
                        
                        if hasattr(chunk, 'delta') and chunk.delta:
                            content = chunk.delta
                        elif hasattr(chunk, 'content') and chunk.content:
                            content = chunk.content
                        elif isinstance(chunk, str):
                            content = chunk
                        
                        if content:
                            chunk_count += 1
                            data = json.dumps({"content": content})
                            yield f"data: {data}\n\n"
                            print(f"📤 Chunk {chunk_count}: {content[:50]}...")
                            await asyncio.sleep(0.01)
                    
                    print(f"✅ Streamed {chunk_count} chunks")
                
                else:
                    raise Exception("Stream object not iterable")
            
            except Exception as stream_error:
                print(f"⚠️ Streaming failed: {str(stream_error)}")
                print("🔄 Falling back to simulated streaming...")
                
                # Fallback: Get full response and simulate streaming
                result = await Runner.run(
                    full_stack_developer_agent,
                    req.message
                )
                full_response = result.final_output
                print(f"📊 Full response length: {len(full_response)}")
                
                # Stream word by word
                words = full_response.split()
                for i, word in enumerate(words):
                    chunk = word if i == 0 else f" {word}"
                    data = json.dumps({"content": chunk})
                    yield f"data: {data}\n\n"
                    await asyncio.sleep(0.04)
                
                print(f"✅ Simulated streaming of {len(words)} words")
            
            # Send done signal
            print("🏁 Sending done signal...")
            yield f"data: {json.dumps({'done': True})}\n\n"
            print(f"{'='*50}\n")
            
        except Exception as e:
            error_msg = str(e)
            print(f"\n❌ ERROR in stream:")
            print(f"📛 Error: {error_msg}")
            traceback.print_exc()
            print(f"{'='*50}\n")
            
            yield f"data: {json.dumps({'error': error_msg})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
