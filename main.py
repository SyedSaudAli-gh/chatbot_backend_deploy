from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
import asyncio

load_dotenv()

# Gemini Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

print("=" * 50)
print("🚀 Starting Chatbot Backend...")
print(f"📍 API Key Present: {bool(GEMINI_API_KEY)}")
print(f"📍 API Key (first 10): {GEMINI_API_KEY[:10] if GEMINI_API_KEY else 'MISSING'}")
print("=" * 50)

if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found in environment!")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# System prompt for Full Stack Developer Agent
SYSTEM_PROMPT = """You are an expert Full Stack Developer Agent specializing in Frontend, Backend, and FastAPI.

Core Skills:
- Frontend: React, Vue, Angular, JavaScript, TypeScript, CSS, Next.js
- Backend: FastAPI (specialist), Python, Node.js, Express
- Databases: SQL (PostgreSQL, MySQL), NoSQL (MongoDB, Redis)
- Tools: Git, Docker, REST APIs, GraphQL

Response Rules:
1. Ask clarifying questions if the request is unclear
2. Provide structured, well-commented code solutions
3. Explain your reasoning and best practices
4. Focus on security, performance, and scalability
5. Use proper code blocks with language tags
6. Be concise but thorough

Your specialty is FastAPI development, frontend-backend integration, and practical problem-solving.
Your goal is to help developers build better software through clear, actionable guidance."""

app = FastAPI(title="Full Stack Developer Agent API")

# CORS Configuration
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
        "endpoints": ["/chat", "/chat/stream", "/health"]
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "api_configured": bool(GEMINI_API_KEY),
        "model": "gemini-2.0-flash-exp"
    }

class ChatRequest(BaseModel):
    message: str

# Non-streaming endpoint
@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        print(f"\n📥 Received: {req.message}")
        
        # Create model
        model = genai.GenerativeModel(
            model_name='gemini-2.0-flash-exp',
            system_instruction=SYSTEM_PROMPT
        )
        
        # Generate response
        response = model.generate_content(req.message)
        
        print(f"✅ Response generated ({len(response.text)} chars)")
        
        return {"response": response.text}
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# Streaming endpoint
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    async def event_generator():
        try:
            print(f"\n{'='*50}")
            print(f"📥 Stream Request: {req.message}")
            
            # Create model
            model = genai.GenerativeModel(
                model_name='gemini-2.0-flash-exp',
                system_instruction=SYSTEM_PROMPT
            )
            
            # Generate streaming response
            response = model.generate_content(
                req.message,
                stream=True
            )
            
            chunk_count = 0
            
            # Stream chunks
            for chunk in response:
                if chunk.text:
                    chunk_count += 1
                    data = json.dumps({"content": chunk.text})
                    yield f"data: {data}\n\n"
                    print(f"📤 Chunk {chunk_count}: {chunk.text[:50]}...")
                    await asyncio.sleep(0.01)
            
            # Send done signal
            yield f"data: {json.dumps({'done': True})}\n\n"
            print(f"✅ Stream completed - {chunk_count} chunks sent")
            print(f"{'='*50}\n")
            
        except Exception as e:
            error_msg = str(e)
            print(f"\n❌ Stream Error: {error_msg}")
            import traceback
            traceback.print_exc()
            
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
