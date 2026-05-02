import os, time, json
import urllib.request as urlrequest
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Tuple, List, Optional

# ==========================================
# CONFIGURATION: Add your Gemini API Key here
# ==========================================
GEMINI_API_KEY = "AIzaSyDeHsitXSRj_Hy-gesBwOzl75VZrLZHAQ0"  # <-- PUT YOUR GEMINI API KEY HERE
GEMINI_MODEL = "gemini-2.5-flash"

def ask_gemini(prompt: str) -> str:
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1500}
    }).encode("utf-8")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    req = urlrequest.Request(url, data=body, headers={"Content-Type": "application/json"})
    resp = urlrequest.urlopen(req, timeout=30)
    data = json.loads(resp.read().decode("utf-8"))
    return data["candidates"][0]["content"]["parts"][0]["text"]

app = FastAPI()
START = time.time()

# In-memory stores
contexts: Dict[Tuple[str, str], dict] = {}    # (scope, context_id) -> {version, payload}
conversations: Dict[str, list] = {}           # conversation_id -> [turns]

@app.get("/")
async def root():
    return {"message": "Vera AI Bot is running. API endpoints are at /v1/"}

@app.get("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _), _ in contexts.items():
        counts[scope] = counts.get(scope, 0) + 1
    return {"status": "ok", "uptime_seconds": int(time.time() - START), "contexts_loaded": counts}

@app.get("/v1/metadata")
async def metadata():
    return {"team_name": "Team AI", "team_members": ["Rohit"], "model": "gemini-2.5-flash",
            "approach": "gemini-llm-integration", "contact_email": "rohit@example.com",
            "version": "2.0.0", "submitted_at": datetime.utcnow().isoformat() + "Z"}

class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: Dict[str, Any]
    delivered_at: str

@app.post("/v1/context")
async def push_context(body: CtxBody):
    key = (body.scope, body.context_id)
    cur = contexts.get(key)
    if cur and cur["version"] > body.version:
        return {"accepted": False, "reason": "stale_version", "current_version": cur["version"]}
    contexts[key] = {"version": body.version, "payload": body.payload}
    return {"accepted": True, "ack_id": f"ack_{body.context_id}_v{body.version}",
            "stored_at": datetime.utcnow().isoformat() + "Z"}

class TickBody(BaseModel):
    now: str
    available_triggers: List[str] = []

@app.post("/v1/tick")
async def tick(body: TickBody):
    actions = []
    
    # If no API key is set, return empty actions safely
    if not GEMINI_API_KEY:
        print("WARN: Gemini API Key not set. Returning empty tick.")
        return {"actions": []}
        
    for trg_id in body.available_triggers:
        trg = contexts.get(("trigger", trg_id), {}).get("payload")
        if not trg: continue
        merchant_id = trg.get("merchant_id")
        merchant = contexts.get(("merchant", merchant_id), {}).get("payload")
        category = contexts.get(("category", merchant.get("category_slug")), {}).get("payload") if merchant else None
        
        if not (merchant and category): continue
        
        # Build prompt for Gemini
        prompt = f"""
        Compose a highly engaging WhatsApp message for a merchant.
        
        Category Context:
        Voice: {category.get('voice')}
        Offers: {category.get('offer_catalog')}
        
        Merchant Context:
        Name: {merchant.get('identity', {}).get('name')}
        Owner: {merchant.get('identity', {}).get('owner_first_name', '')}
        Languages: {merchant.get('identity', {}).get('languages', ['en'])}
        Performance: {merchant.get('performance')}
        
        Trigger Context:
        Kind: {trg.get('kind')}
        Payload: {trg}
        
        Rules:
        1. Keep it short and engaging.
        2. Use numbers and concrete specifics from the contexts to anchor the message.
        3. Only end with a clear YES/STOP binary CTA if action is needed.
        
        Return ONLY a JSON response in this exact format, with no markdown code blocks:
        {{"body": "The message text", "cta": "YES/STOP or open_ended", "rationale": "Why you wrote this"}}
        """
        
        try:
            text_response = ask_gemini(prompt)
            # Clean up response if it contains markdown formatting
            text_response = text_response.replace("```json", "").replace("```", "").strip()
            result = json.loads(text_response)
            
            actions.append({
                "conversation_id": f"conv_{merchant_id}_{trg_id}",
                "merchant_id": merchant_id, "customer_id": None,
                "send_as": "vera", "trigger_id": trg_id,
                "template_name": "vera_generic_v1",
                "template_params": [merchant.get('identity', {}).get('owner_first_name', 'Merchant'), trg_id, "details"],
                "body": result.get("body", "Hello!"), "cta": result.get("cta", "YES/STOP"),
                "suppression_key": trg.get("suppression_key", ""),
                "rationale": result.get("rationale", "Composed via Gemini")
            })
        except Exception as e:
            print(f"Error generating content for {merchant_id}: {e}")
            
    return {"actions": actions}

class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int

@app.post("/v1/reply")
async def reply(body: ReplyBody):
    turns = conversations.setdefault(body.conversation_id, [])
    turns.append({"from": body.from_role, "msg": body.message})
    
    if not GEMINI_API_KEY:
        print("WARN: Gemini API Key not set. Falling back to heuristic reply.")
        merchant_msgs = [t["msg"] for t in turns if t["from"] == "merchant"]
        if len(merchant_msgs) >= 3 and merchant_msgs[-1] == merchant_msgs[-2] == merchant_msgs[-3]:
            return {"action": "end", "rationale": "Detected auto-reply loop."}
        if "stop" in body.message.lower():
            return {"action": "end", "body": "I apologize. I won't message again.", "cta": "none", "rationale": "Hostile."}
        return {"action": "send", "body": "Got it. I will process that for you.", "cta": "none", "rationale": "Default"}

    history_str = "\\n".join([f"{t['from'].capitalize()}: {t['msg']}" for t in turns])
    
    prompt = f"""
    You are an AI assistant evaluating a conversation with a merchant.
    Determine the next action based on this history:
    
    {history_str}
    
    Rules:
    1. If the merchant's last message is exactly the same as their previous two messages, it is an auto-reply loop. Action: "end".
    2. If the merchant is hostile (angry, says "stop", "spam"), apologize gracefully. Action: "end".
    3. If the merchant clearly agreed (e.g. "ok let's do it"), switch to action mode and confirm task is done. Action: "send".
    4. Otherwise, continue the conversation politely. Action: "send" or "wait".
    
    Return ONLY a JSON response in this exact format, with no markdown code blocks:
    {{"action": "send|wait|end", "body": "Response text (if sending)", "cta": "open_ended|none", "rationale": "Why"}}
    """
    
    try:
        text_response = ask_gemini(prompt)
        text_response = text_response.replace("```json", "").replace("```", "").strip()
        result = json.loads(text_response)
        
        return {
            "action": result.get("action", "end"),
            "body": result.get("body", "Got it."),
            "cta": result.get("cta", "none"),
            "rationale": result.get("rationale", "Decided by Gemini")
        }
    except Exception as e:
        print(f"Error handling reply: {e}")
        return {"action": "end", "rationale": "Error generating response"}
