from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import whisper
import torch
import tempfile
import os
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading Qwen2.5-VL model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

print("Loading Whisper model...")
whisper_model = whisper.load_model("tiny")

print("All models loaded!")


@app.get("/")
def root():
    return {
        "status": "Chaka Model API is running",
        "model": "Qwen2.5-VL-7B-Instruct + Whisper-tiny",
        "endpoints": ["/chat", "/health"]
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat(
    message: str = Form(...),
    image: UploadFile = File(None),
    audio: UploadFile = File(None)
):
    content = []
    tmp_img = None

    # Handle audio — transcribe with Whisper first
    if audio and audio.filename:
        audio_bytes = await audio.read()
        suffix = os.path.splitext(audio.filename)[-1] or ".mp3"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(audio_bytes)
            tmp_audio = f.name
        try:
            result = whisper_model.transcribe(tmp_audio)
            transcribed = result.get("text", "").strip()
            if transcribed:
                content.append({
                    "type": "text",
                    "text": f"[Voice note]: {transcribed}"
                })
        finally:
            os.unlink(tmp_audio)

    # Handle image
    if image and image.filename:
        img_bytes = await image.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
            img.save(f.name)
            tmp_img = f.name
        content.append({"type": "image", "image": tmp_img})

    # Add text message
    content.append({"type": "text", "text": message})

    messages = [{"role": "user", "content": content}]

    # Build input
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    # Generate response
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=1024)

    generated = output_ids[0][inputs.input_ids.shape[1]:]
    response = processor.decode(generated, skip_special_tokens=True)

    # Cleanup
    if tmp_img:
        try:
            os.unlink(tmp_img)
        except Exception:
            pass

    return {"response": response}
