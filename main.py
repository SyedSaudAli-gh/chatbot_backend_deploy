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

# Environment Variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")

# Startup Logs
print("=" * 60)
print("🚀 Starting Full Stack Developer Agent Backend...")
print(f"📍 API Key Present: {bool(GEMINI_API_KEY)}")
if GEMINI_API_KEY:
    print(f"📍 API Key (first 10): {GEMINI_API_KEY[:10]}")
print(f"📍 Base URL: {GEMINI_BASE_URL}")
print("=" * 60)

# Validation
if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found in environment variables!")

# Initialize OpenAI Client for Gemini
client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url=GEMINI_BASE_URL
)

# Initialize Model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash-exp",
    openai_client=client
)

# Create Agent
full_stack_developer_agent = Agent(
    name="Full Stack Developer Agent",
    instructions="""
        Role: Expert Full Stack Developer specializing in Frontend, Backend, and FastAPI.

        Core Skills:
        - Frontend: React, Vue, Angular, JavaScript, TypeScript, CSS, Next.js, Tailwind
        - Backend: FastAPI (specialist), Python, Node.js, Express
        - Databases: SQL (PostgreSQL, MySQL), NoSQL (MongoDB, Redis)
        - Tools: Git, Docker, REST APIs, GraphQL, CI/CD

        Response Rules:
        - Ask clarifying questions if request is unclear
        - Provide structured, well-commented code solutions
        - Explain reasoning behind answers with examples
        - Focus on security, performance, and best practices
        - Use proper code blocks with language tags
        - Be concise yet thorough

        Specialty: FastAPI development, frontend-backend integration, problem-solving.
        Goal: Help users build better software through clear, actionable guidance.
    """,
    model=model
)

app = FastAPI(title="Full Stack Developer Agent API")

# CORS Middleware
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
        "message": "Hello from Saud Ali - Full Stack Developer Agent",
        "status": "running",
        "model": "gemini-2.0-flash-exp",
        "api_configured": bool(GEMINI_API_KEY),
        "endpoints": ["/chat", "/chat/stream", "/health"]
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "api_configured": bool(GEMINI_API_KEY),
        "base_url": GEMINI_BASE_URL,
        "model": "gemini-2.0-flash-exp"
    }

class ChatRequest(BaseModel):
    message: str

# Non-streaming endpoint
@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        print(f"\n📥 Received message: {req.message}")
        
        result = await Runner.run(
            full_stack_developer_agent,
            req.message
        )
        
        print(f"✅ Response generated: {len(result.final_output)} chars")
        
        return {"response": result.final_output}
    
    except Exception as e:
        print(f"❌ Error in /chat: {str(e)}")
        traceback.print_exc()
        return {"error": str(e)}

# Streaming endpoint
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    async def event_generator():
        try:
            print(f"\n{'='*60}")
            print(f"📥 Stream Request: {req.message}")
            print(f"🔄 Starting streaming response...")
            
            # Try direct streaming first
            try:
                print("🔍 Attempting Runner.run_streamed...")
                stream_result = Runner.run_streamed(
                    full_stack_developer_agent,
                    req.message
                )
                
                print(f"📊 Stream result type: {type(stream_result)}")
                
                chunk_count = 0
                streamed_successfully = False
                
                # Method 1: Check for .stream() method
                if hasattr(stream_result, 'stream'):
                    print("✅ Using .stream() method")
                    async for chunk in stream_result.stream():
                        content = None
                        
                        # Try different chunk formats
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
                            
                            # Log first 50 chars
                            preview = content[:50] if len(content) > 50 else content
                            print(f"📤 Chunk {chunk_count}: {preview}...")
                            
                            await asyncio.sleep(0.01)
                            streamed_successfully = True
                    
                    print(f"✅ Streamed {chunk_count} chunks using .stream()")
                
                # Method 2: Direct async iteration
                elif hasattr(stream_result, '__aiter__'):
                    print("✅ Using async iteration")
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
                            
                            preview = content[:50] if len(content) > 50 else content
                            print(f"📤 Chunk {chunk_count}: {preview}...")
                            
                            await asyncio.sleep(0.01)
                            streamed_successfully = True
                    
                    print(f"✅ Streamed {chunk_count} chunks using async iteration")
                
                # If no streaming worked, fall back
                if not streamed_successfully:
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
                print(f"📊 Full response length: {len(full_response)} chars")
                
                # Stream word by word for better UX
                words = full_response.split()
                for i, word in enumerate(words):
                    chunk = word if i == 0 else f" {word}"
                    data = json.dumps({"content": chunk})
                    yield f"data: {data}\n\n"
                    await asyncio.sleep(0.03)  # Slightly faster
                
                print(f"✅ Simulated streaming of {len(words)} words")
            
            # Send done signal
            print("🏁 Sending done signal...")
            yield f"data: {json.dumps({'done': True})}\n\n"
            print(f"{'='*60}\n")
            
        except Exception as e:
            error_msg = str(e)
            print(f"\n❌ ERROR in stream:")
            print(f"📛 Error: {error_msg}")
            traceback.print_exc()
            print(f"{'='*60}\n")
            
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
