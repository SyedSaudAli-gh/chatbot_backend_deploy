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

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL")

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
    return {"message": "Hello from Saud Ali"}

class ChatRequest(BaseModel):
    message: str

# Non-streaming endpoint
@app.post("/chat")
async def chat(req: ChatRequest):
    result = await Runner.run(
        full_stack_developer_agent,
        req.message
    )
    return {"response": result.final_output}

# Streaming endpoint - FIXED VERSION
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    async def event_generator():
        try:
            print(f"Received message: {req.message}")
            
            # Run streaming
            stream_result = Runner.run_streamed(
                full_stack_developer_agent,
                req.message
            )
            
            print(f"Stream result type: {type(stream_result)}")
            
            # Check if result has streaming attribute
            if hasattr(stream_result, 'stream'):
                print("Using .stream() method")
                async for chunk in stream_result.stream():
                    if hasattr(chunk, 'delta') and chunk.delta:
                        content = chunk.delta
                    elif hasattr(chunk, 'content') and chunk.content:
                        content = chunk.content
                    else:
                        continue
                    
                    data = json.dumps({"content": content})
                    yield f"data: {data}\n\n"
                    print(f"Sent chunk: {content[:50]}...")
                    await asyncio.sleep(0.01)
            
            # Check if it's directly iterable
            elif hasattr(stream_result, '__aiter__'):
                print("Using async iteration")
                async for chunk in stream_result:
                    if hasattr(chunk, 'delta') and chunk.delta:
                        content = chunk.delta
                    elif hasattr(chunk, 'content') and chunk.content:
                        content = chunk.content
                    else:
                        continue
                    
                    data = json.dumps({"content": content})
                    yield f"data: {data}\n\n"
                    print(f"Sent chunk: {content[:50]}...")
                    await asyncio.sleep(0.01)
            
            # Fallback: Get full response and simulate streaming
            else:
                print("Fallback: Using simulated streaming")
                result = await Runner.run(
                    full_stack_developer_agent,
                    req.message
                )
                full_response = result.final_output
                print(f"Got response length: {len(full_response)}")
                
                # Stream word by word
                words = full_response.split()
                for i, word in enumerate(words):
                    chunk = word if i == 0 else f" {word}"
                    data = json.dumps({"content": chunk})
                    yield f"data: {data}\n\n"
                    await asyncio.sleep(0.04)
            
            # Send done signal
            yield f"data: {json.dumps({'done': True})}\n\n"
            print("Stream completed")
            
        except Exception as e:
            error_msg = str(e)
            print(f"Error: {error_msg}")
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