"""
Facebook Auto-Poster Bot  (text+image auto-queue  +  text+video on-demand)
==========================================================================
• Upload posts (text or photo+caption) -> queued -> every POST_INTERVAL (default
  6h) the bot rewrites with DeepSeek, makes a DALL-E image, and posts to your Page.
• 🎬 Generate + Preview: send content -> DeepSeek rewrite -> Veo (Gemini API) 8s
  video -> you get a preview with Approve/Discard -> posts the VIDEO only if approved.

Buttons:  ▶️ Start   ⏹ Stop   🎬 Generate + Preview   📊 Queue   🗑 Clear
All secrets come from environment variables — nothing is hardcoded.
"""

import os
import io
import sys
import json
import time
import uuid
import base64
import asyncio
import logging

import requests
from telegram import (
    Update, ReplyKeyboardMarkup, InlineKeyboardMarkup, InlineKeyboardButton,
)
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler,
    filters, ContextTypes,
)

print(">>> SYSTEM: SCRIPT STARTING UP...", flush=True)

# --------------------------------------------------------------------------- #
#  Configuration (everything from env — NO hardcoded secrets)
# --------------------------------------------------------------------------- #
TELEGRAM_TOKEN        = os.getenv("TELEGRAM_TOKEN")
FB_PAGE_ID            = os.getenv("FB_PAGE_ID")
FB_PAGE_ACCESS_TOKEN  = os.getenv("FB_PAGE_ACCESS_TOKEN")
DEEPSEEK_API_KEY      = os.getenv("DEEPSEEK_API_KEY")
OPENAI_API_KEY        = os.getenv("OPENAI_API_KEY")          # DALL-E images (auto-queue)
GOOGLE_API_KEY        = os.getenv("GOOGLE_API_KEY")          # Gemini/Veo (videos)
VEO_MODEL             = os.getenv("VEO_MODEL", "veo-3.0-fast-generate-001")
POST_INTERVAL         = int(os.getenv("POST_INTERVAL", "21600"))   # 6h default
QUEUE_FILE            = os.getenv("QUEUE_FILE", "queue.json")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO, stream=sys.stdout,
)
logger = logging.getLogger("fbposter")

for label, val in [
    ("TELEGRAM_TOKEN", TELEGRAM_TOKEN), ("FB_PAGE_ID", FB_PAGE_ID),
    ("FB_PAGE_ACCESS_TOKEN", FB_PAGE_ACCESS_TOKEN), ("DEEPSEEK_API_KEY", DEEPSEEK_API_KEY),
    ("OPENAI_API_KEY", OPENAI_API_KEY), ("GOOGLE_API_KEY", GOOGLE_API_KEY),
]:
    print(f">>> CONFIG {label}: {'SET' if val else 'MISSING'}", flush=True)
print(f">>> CONFIG VEO_MODEL: {VEO_MODEL} | POST_INTERVAL: {POST_INTERVAL}s", flush=True)


# --------------------------------------------------------------------------- #
#  State + persistent queue (survives restarts)
# --------------------------------------------------------------------------- #
class BotState:
    def __init__(self):
        d = self._load()
        self.queue = d.get("queue", [])
        self.owner_chat_id = d.get("owner_chat_id")
        self.is_running = d.get("is_running", False)
        self.awaiting_video = False         # waiting for content for a video preview
        self.pending_video = None           # {"text":..., "video":bytes} awaiting approval

    def _load(self):
        try:
            with open(QUEUE_FILE) as f:
                d = json.load(f)
            return d if isinstance(d, dict) else {"queue": d}
        except Exception:
            return {}

    def save(self):
        try:
            with open(QUEUE_FILE, "w") as f:
                json.dump({"queue": self.queue, "owner_chat_id": self.owner_chat_id,
                           "is_running": self.is_running}, f)
        except Exception as e:
            logger.warning("Could not save state: %s", e)

    def add(self, text, image_b64=None):
        self.queue.append({"id": uuid.uuid4().hex[:8], "text": text, "image_b64": image_b64})
        self.save()
        return len(self.queue)


state = BotState()


# --------------------------------------------------------------------------- #
#  DeepSeek (rewrite + media prompts)
# --------------------------------------------------------------------------- #
REWRITE_SYS = (
    "You rewrite social-media posts into fresh, original copy so they are not "
    "duplicate content. Keep the core message, facts and tone, but reword it so it is "
    "clearly unique. Concise and engaging. You may add relevant emojis and 2-4 hashtags. "
    "Output ONLY the rewritten post text — no preamble, no quotes."
)
IMG_SYS = (
    "Write ONE concise, safe-for-work prompt for the DALL-E 3 image generator that "
    "visually represents the following post. No real people's faces, no brands, no "
    "text/letters in the image. Output ONLY the prompt, max 55 words."
)
VIDEO_SYS = (
    "Write ONE concise prompt for an AI text-to-video generator (an 8-second clip). "
    "Describe a single dynamic scene with clear motion that represents the post. "
    "No on-screen text/letters, no real people's faces, no brands. "
    "Output ONLY the prompt, max 60 words."
)


def deepseek_chat(system, user, temperature=1.3):
    """Returns (content, error)."""
    if not DEEPSEEK_API_KEY:
        return None, "DEEPSEEK_API_KEY not set."
    try:
        r = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                     "Content-Type": "application/json"},
            json={"model": "deepseek-chat",
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user}],
                  "temperature": temperature, "stream": False},
            timeout=60,
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip(), None
        return None, f"DeepSeek {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return None, f"DeepSeek error: {e}"


# --------------------------------------------------------------------------- #
#  OpenAI DALL-E 3 image  (used by the auto-queue)
# --------------------------------------------------------------------------- #
def generate_image_dalle(prompt):
    """Returns (image_bytes, error). dall-e-3 returns a URL by default (response_format
    is no longer accepted), so we download it; b64_json kept as a fallback."""
    if not OPENAI_API_KEY:
        return None, "OPENAI_API_KEY not set."
    try:
        r = requests.post(
            "https://api.openai.com/v1/images/generations",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}",
                     "Content-Type": "application/json"},
            json={"model": "dall-e-3", "prompt": prompt[:3900], "n": 1,
                  "size": "1024x1024"},
            timeout=120,
        )
        if r.status_code != 200:
            return None, f"DALL-E {r.status_code}: {r.text[:200]}"
        data = r.json().get("data", [{}])[0]
        if data.get("b64_json"):
            return base64.b64decode(data["b64_json"]), None
        if data.get("url"):
            img = requests.get(data["url"], timeout=60)
            if img.status_code == 200:
                return img.content, None
            return None, f"image download failed: HTTP {img.status_code}"
        return None, f"DALL-E returned no image: {r.text[:200]}"
    except Exception as e:
        return None, f"DALL-E error: {e}"


# --------------------------------------------------------------------------- #
#  Google Veo video (Gemini API).  ⚠️ ~$3-6 per 8s clip — generated on demand only.
#  NOTE: Veo SDK shapes vary by version; verify on the first real run.
# --------------------------------------------------------------------------- #
def generate_video_veo(prompt):
    """Returns (video_bytes, error). Long-running (1-3 min)."""
    if not GOOGLE_API_KEY:
        return None, "GOOGLE_API_KEY not set."
    try:
        from google import genai
        from google.genai import types
    except Exception as e:
        return None, f"google-genai not installed: {e}"
    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        operation = client.models.generate_videos(
            model=VEO_MODEL,
            prompt=prompt,
            config=types.GenerateVideosConfig(aspect_ratio="16:9"),
        )
        waited = 0
        while not operation.done and waited < 360:
            time.sleep(15)
            waited += 15
            operation = client.operations.get(operation)
        if not operation.done:
            return None, "Veo timed out (>6 min)."
        vids = getattr(operation.response, "generated_videos", None) or []
        if not vids:
            return None, f"Veo returned no video ({operation.response})."
        video = vids[0].video
        client.files.download(file=video)           # populates bytes
        data = getattr(video, "video_bytes", None)
        if not data:
            return None, "Veo produced a video but no bytes could be downloaded."
        return data, None
    except Exception as e:
        return None, f"Veo error: {e}"


# --------------------------------------------------------------------------- #
#  Facebook posting (photo/feed + video) + link helpers
# --------------------------------------------------------------------------- #
def post_to_fb(text, image_bytes=None):
    if not FB_PAGE_ID:
        return {"error": {"message": "FB_PAGE_ID missing"}}
    if not FB_PAGE_ACCESS_TOKEN:
        return {"error": {"message": "FB_PAGE_ACCESS_TOKEN missing"}}
    base = f"https://graph.facebook.com/v21.0/{FB_PAGE_ID}"
    try:
        if image_bytes:
            files = {"source": ("image.png", io.BytesIO(image_bytes), "image/png")}
            data = {"access_token": FB_PAGE_ACCESS_TOKEN, "message": text, "published": "true"}
            res = requests.post(f"{base}/photos", data=data, files=files, timeout=60)
        else:
            data = {"access_token": FB_PAGE_ACCESS_TOKEN, "message": text}
            res = requests.post(f"{base}/feed", data=data, timeout=60)
        return res.json()
    except Exception as e:
        return {"error": {"message": str(e)}}


def post_video_to_fb(text, video_bytes):
    if not FB_PAGE_ID or not FB_PAGE_ACCESS_TOKEN:
        return {"error": {"message": "FB_PAGE_ID / FB_PAGE_ACCESS_TOKEN missing"}}
    try:
        url = f"https://graph-video.facebook.com/v21.0/{FB_PAGE_ID}/videos"
        files = {"source": ("video.mp4", io.BytesIO(video_bytes), "video/mp4")}
        data = {"access_token": FB_PAGE_ACCESS_TOKEN, "description": text}
        res = requests.post(url, data=data, files=files, timeout=300)
        return res.json()
    except Exception as e:
        return {"error": {"message": str(e)}}


def fb_post_link(res):
    pid = res.get("post_id") or res.get("id")
    return f"https://www.facebook.com/{pid}" if pid else None


def _token_hint(msg, res):
    if "expired" in msg.lower() or res.get("error", {}).get("code") == 190:
        return msg + "\n\n👉 FB_PAGE_ACCESS_TOKEN expired — generate a fresh PAGE token."
    if "publish_actions" in msg or "pages_manage_posts" in msg:
        return msg + "\n\n👉 You're using a USER token. Use a PAGE token with 'pages_manage_posts'."
    return msg


# --------------------------------------------------------------------------- #
#  Auto-queue: post the next item as text + DALL-E image
# --------------------------------------------------------------------------- #
async def post_next(bot, chat_id):
    if not state.queue:
        await bot.send_message(chat_id=chat_id, text="📭 Queue is empty — send me some posts to add.")
        return
    item = state.queue[0]
    await bot.send_message(chat_id=chat_id, text="⏳ Preparing next post…")

    rewritten, err = await asyncio.to_thread(deepseek_chat, REWRITE_SYS, item["text"], 1.3)
    if not rewritten:
        await bot.send_message(chat_id=chat_id,
                               text=f"⚠️ DeepSeek failed ({err}); keeping in queue, will retry next cycle.")
        return

    if item.get("image_b64"):
        img_bytes = base64.b64decode(item["image_b64"])
    else:
        prompt, _ = await asyncio.to_thread(deepseek_chat, IMG_SYS, rewritten[:500], 1.0)
        img_bytes, img_err = await asyncio.to_thread(
            generate_image_dalle, prompt or f"A clean, professional illustration about: {rewritten[:120]}")
        if not img_bytes:
            await bot.send_message(chat_id=chat_id, text=f"⚠️ Image step failed ({img_err}); posting text-only.")

    res = await asyncio.to_thread(post_to_fb, rewritten, img_bytes)
    if "id" in res or "post_id" in res:
        state.queue.pop(0)
        state.save()
        link = fb_post_link(res)
        cap = f"✅ Posted!\n\n{rewritten}\n\n🔗 {link or res.get('id')}\n📊 {len(state.queue)} left in queue"
        if img_bytes:
            try:
                await bot.send_photo(chat_id=chat_id, photo=io.BytesIO(img_bytes), caption=cap[:1024])
            except Exception:
                await bot.send_message(chat_id=chat_id, text=cap)
        else:
            await bot.send_message(chat_id=chat_id, text=cap)
        if not state.queue:
            await bot.send_message(chat_id=chat_id, text="🏁 Queue empty — send me more anytime.")
    else:
        msg = _token_hint(res.get("error", {}).get("message", "Unknown error"), res)
        await bot.send_message(chat_id=chat_id, text=f"❌ Facebook post failed (item kept in queue):\n{msg}")


async def post_job(context: ContextTypes.DEFAULT_TYPE):
    await post_next(context.bot, context.job.chat_id)


# --------------------------------------------------------------------------- #
#  On-demand text+video: generate -> preview -> approve/discard
# --------------------------------------------------------------------------- #
async def make_video_preview(bot, chat_id, content):
    await bot.send_message(chat_id=chat_id, text="⏳ Rewriting + generating the video (Veo, ~1-3 min)…")
    rewritten, err = await asyncio.to_thread(deepseek_chat, REWRITE_SYS, content, 1.3)
    if not rewritten:
        await bot.send_message(chat_id=chat_id, text=f"❌ DeepSeek rewrite failed: {err}")
        return
    vprompt, _ = await asyncio.to_thread(deepseek_chat, VIDEO_SYS, rewritten[:500], 1.0)
    video, verr = await asyncio.to_thread(generate_video_veo, vprompt or rewritten[:120])
    if not video:
        await bot.send_message(chat_id=chat_id, text=f"❌ Video generation failed:\n{verr}")
        return
    state.pending_video = {"text": rewritten, "video": video}
    kb = InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ Approve & Post", callback_data="approve"),
        InlineKeyboardButton("❌ Discard", callback_data="discard"),
    ]])
    try:
        await bot.send_video(chat_id=chat_id, video=io.BytesIO(video),
                             caption=f"🎬 Preview — approve to post:\n\n{rewritten[:900]}", reply_markup=kb)
    except Exception as e:
        await bot.send_message(chat_id=chat_id,
                               text=f"(Couldn't send video preview: {e})\n\nText:\n{rewritten}", reply_markup=kb)


async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    chat_id = q.message.chat_id
    if q.data == "discard":
        state.pending_video = None
        await context.bot.send_message(chat_id=chat_id, text="❌ Discarded — nothing posted.")
        return
    if q.data == "approve":
        pv = state.pending_video
        if not pv:
            await context.bot.send_message(chat_id=chat_id, text="Nothing pending to post.")
            return
        await context.bot.send_message(chat_id=chat_id, text="📤 Posting video to Facebook…")
        res = await asyncio.to_thread(post_video_to_fb, pv["text"], pv["video"])
        if "id" in res:
            link = f"https://www.facebook.com/{FB_PAGE_ID}/videos/{res['id']}"
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"✅ Video posted! (Facebook may take a minute to finish processing.)\n🔗 {link}")
        else:
            msg = _token_hint(res.get("error", {}).get("message", "Unknown error"), res)
            await context.bot.send_message(chat_id=chat_id, text=f"❌ FB video post failed:\n{msg}")
        state.pending_video = None


# --------------------------------------------------------------------------- #
#  Telegram UI
# --------------------------------------------------------------------------- #
KEYBOARD = ReplyKeyboardMarkup(
    [["▶️ Start", "⏹ Stop"], ["🎬 Generate + Preview"], ["📊 Queue", "🗑 Clear"]],
    resize_keyboard=True,
)


def status_text():
    return (f"📊 Queue: {len(state.queue)} post(s) waiting\n"
            f"Auto-posting (text+image): {'🟢 ON' if state.is_running else '🔴 off'} "
            f"(every {POST_INTERVAL//3600}h)")


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state.owner_chat_id = update.effective_chat.id
    await update.message.reply_text(
        "🤖 Facebook Auto-Poster\n\n"
        "📝 Send me posts (text or photo+caption) → queued. Tap ▶️ Start to auto-post one "
        f"every {POST_INTERVAL//3600}h with a DALL-E image.\n\n"
        "🎬 Generate + Preview → send content, I'll make an 8s Veo video and show you a "
        "preview to Approve or Discard before it posts.\n\n"
        + status_text(),
        reply_markup=KEYBOARD,
    )


def _start_posting(context, chat_id):
    for j in context.job_queue.get_jobs_by_name("poster"):
        j.schedule_removal()
    context.job_queue.run_repeating(post_job, interval=POST_INTERVAL, first=5, name="poster", chat_id=chat_id)
    state.is_running = True
    state.owner_chat_id = chat_id
    state.save()


def _stop_posting(context):
    for j in context.job_queue.get_jobs_by_name("poster"):
        j.schedule_removal()
    state.is_running = False
    state.save()


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cmd = (update.message.text or "").strip()
    chat_id = update.effective_chat.id
    state.owner_chat_id = chat_id

    if cmd == "🎬 Generate + Preview":
        state.awaiting_video = True
        await update.message.reply_text("🎬 Send me the content for your text+video post.")
        return

    if cmd == "▶️ Start":
        state.awaiting_video = False
        if not state.queue:
            await update.message.reply_text("📭 Queue is empty — send me some posts first.")
            return
        _start_posting(context, chat_id)
        await update.message.reply_text(
            f"🟢 Started! First post in a few seconds, then one every {POST_INTERVAL//3600}h.")
        return

    if cmd == "⏹ Stop":
        state.awaiting_video = False
        _stop_posting(context)
        await update.message.reply_text("🔴 Auto-posting stopped. Your queue is saved.")
        return

    if cmd in ("📊 Queue", "/status", "Status"):
        await update.message.reply_text(status_text())
        return

    if cmd == "🗑 Clear":
        state.queue = []
        state.save()
        await update.message.reply_text("🗑 Queue cleared.")
        return

    # Non-button text:
    if state.awaiting_video:
        state.awaiting_video = False
        await make_video_preview(context.bot, chat_id, cmd)
    else:
        n = state.add(cmd)
        await update.message.reply_text(
            f"✅ Queued as post #{n}. Send the next 👇 (or ▶️ Start to begin posting)")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    state.owner_chat_id = chat_id
    photo = await update.message.photo[-1].get_file()
    raw = await photo.download_as_bytearray()
    caption = update.message.caption or ""
    if not caption.strip():
        await update.message.reply_text("📸 Got the photo — resend it *with a caption* so I have text to post.")
        return
    n = state.add(caption, image_b64=base64.b64encode(bytes(raw)).decode("utf-8"))
    await update.message.reply_text(f"✅ Queued post #{n} with your photo. Send the next 👇")


# --------------------------------------------------------------------------- #
#  Startup
# --------------------------------------------------------------------------- #
def verify_fb_token():
    if not FB_PAGE_ACCESS_TOKEN:
        print(">>> FB: no page token set", flush=True)
        return
    try:
        r = requests.get("https://graph.facebook.com/v21.0/me",
                         params={"access_token": FB_PAGE_ACCESS_TOKEN}, timeout=20)
        d = r.json()
        if "name" in d:
            print(f">>> FB: token OK — acting as {d.get('name')} (id {d.get('id')})", flush=True)
        else:
            print(f">>> FB WARNING: token check returned: {d}", flush=True)
    except Exception as e:
        print(f">>> FB: token check failed: {e}", flush=True)


async def _post_init(app: Application):
    if state.is_running and state.owner_chat_id:
        app.job_queue.run_repeating(post_job, interval=POST_INTERVAL, first=10,
                                    name="poster", chat_id=state.owner_chat_id)
        print(f">>> SYSTEM: resumed auto-posting for chat {state.owner_chat_id}", flush=True)


def main():
    if not TELEGRAM_TOKEN:
        print("CRITICAL: TELEGRAM_TOKEN not set — cannot start.", flush=True)
        return
    verify_fb_token()
    app = Application.builder().token(TELEGRAM_TOKEN).post_init(_post_init).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", handle_text))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    print(">>> SYSTEM: BOT POLLING STARTED", flush=True)
    app.run_polling()


if __name__ == "__main__":
    main()
