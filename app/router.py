import os
from fastapi import HTTPException, APIRouter
from model.model import ChatRequest, ChatResponse
from Prompt.build_prompt import (
    _build_system_prompt,
    _extract_json,
    _extract_recommendations_from_text,
    _validate_recommendations,
)
from dotenv import load_dotenv
router = APIRouter()

def _gemini_contents(messages):
    contents = []
    for message in messages:
        role = "model" if message["role"] == "assistant" else "user"
        contents.append({
            "role": role,
            "parts": [{"text": message["content"]}],
        })
    return contents


def _call_gemini(system: str, messages: list[dict]) -> str:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is missing",
        )

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="Gemini SDK missing. Run: uv sync",
        ) from exc

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=_gemini_contents(messages),
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.2,
            max_output_tokens=1024,
            response_mime_type="application/json",
        ),
    )
    return response.text or ""

@router.get("/health")
def health():
    return {"status": "ok"}

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    history_text = "\n".join(
        f"{'USER' if m.role == 'user' else 'ASSISTANT'}: {m.content}"
        for m in req.messages
    )

    # Check turn cap (8 total messages = 4 rounds)
    if len(req.messages) >= 8:
        system = _build_system_prompt(history_text) + "\n\nIMPORTANT: This is the final turn. You MUST provide your best shortlist now and set end_of_conversation to true."
    else:
        system = _build_system_prompt(history_text)
    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    try:
        raw_text = _call_gemini(system, messages)
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")

    # Parse structured JSON from response
    parsed = _extract_json(raw_text)

    if parsed:
        reply = parsed.get("reply", raw_text)
        raw_recs = parsed.get("recommendations", [])
        eoc = bool(parsed.get("end_of_conversation", False))
    else:
        reply = raw_text
        raw_recs = []
        eoc = False

    recommendations = _validate_recommendations(raw_recs)
    if not recommendations:
        recommendations = _extract_recommendations_from_text(f"{reply}\n{raw_text}")

    return ChatResponse(
        reply=reply,
        recommendations=recommendations,
        end_of_conversation=eoc,
    )
