import json
from io import BytesIO
from datetime import datetime

import ollama
import streamlit as st


DEFAULT_MODEL = "llama3.2:3b"
IMAGE_MODEL = "runwayml/stable-diffusion-v1-5"
SYSTEM = "You are a helpful, detailed, and friendly AI assistant. Use clear structure and practical examples."

st.set_page_config(page_title="JARVIS AI", page_icon=":material/auto_awesome:", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink:#e8eef8; --muted:#a8b6ca; --line:#34445e; --mint:#1d3150; --green:#78b7ff; --paper:#0d1422; --white:#182337; }
    * { font-family:'DM Sans', sans-serif; }
    html, body {
        background:#0d1422 !important;
        min-height:100vh !important;
        color:var(--ink) !important;
    }
    .stApp::before {
        content:"";
        position:fixed;
        inset:0;
        z-index:0;
        pointer-events:none;
        background:
            linear-gradient(rgba(7,12,24,.82), rgba(10,18,34,.9)),
            url("https://images.unsplash.com/photo-1519608487953-e999c86e7455?auto=format&fit=crop&w=2400&q=85")
            center / cover no-repeat;
    }
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stAppViewBlockContainer"], .main, section.main {
        background:transparent !important;
        min-height:100vh !important;
    }
    .stApp > * { position:relative; z-index:1; }
    [data-testid="stHeader"] { background:transparent !important; }
    [data-testid="stSidebar"], [data-testid="stSidebarContent"] { background:#0f1929 !important; border-right:1px solid var(--line); }
    [data-testid="stSidebar"] > div:first-child { padding:1.4rem 1.1rem; }
    [data-testid="stSidebar"] * { color:var(--ink); }
    [data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] small, [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:var(--muted) !important; }
    .brand { display:flex; align-items:center; gap:.65rem; margin-bottom:1.5rem; }
    .brand-mark { background:#dceaff; color:#14233a; width:34px; height:34px; border-radius:11px; display:grid; place-items:center; font:700 17px 'Space Grotesk'; }
    .brand-name { color:var(--ink) !important; font:700 20px 'Space Grotesk'; letter-spacing:-.04em; }
    .brand-sub { color:var(--muted); font-size:11px; margin-top:-4px; }
    .section-label { color:#8bbcf0 !important; font-size:10px; font-weight:700; letter-spacing:.12em; text-transform:uppercase; margin:1.6rem 0 .55rem; }
    .hero { max-width:820px; margin:0 auto; padding:4.2rem 1rem 2rem; }
    .eyebrow { color:var(--green) !important; font-size:12px; font-weight:700; letter-spacing:.12em; text-transform:uppercase; margin-bottom:1rem; }
    .hero h1 { font:700 clamp(2.3rem,5vw,4rem)/1.02 'Space Grotesk'; letter-spacing:-.065em; margin:0; color:var(--ink); }
    .hero h1 span { color:#8bbcf0 !important; }
    .hero p { color:var(--muted) !important; font-size:16px; margin:1rem 0 2.2rem; }
    .prompt-card { background:rgba(24,35,55,.9); border:1px solid var(--line); border-radius:14px; padding:1rem 1.1rem; margin-bottom:.7rem; box-shadow:0 8px 24px rgba(0,0,0,.22); }
    .prompt-card strong { color:var(--ink) !important; font-size:14px; }
    .prompt-card small { display:block; color:var(--muted) !important; margin-top:4px; }
    .chat-wrap { max-width:820px; margin:0 auto; padding:2.4rem 1rem 8rem; }
    .chat-title { display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--line); padding-bottom:1rem; margin-bottom:1.4rem; }
    .chat-title h2 { color:var(--ink) !important; font:600 18px 'Space Grotesk'; margin:0; letter-spacing:-.03em; }
    [data-testid="stChatMessage"] { background:transparent; color:var(--ink) !important; padding:1rem 0; }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"], [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p { color:var(--ink) !important; font-size:15px; line-height:1.7; }
    [data-testid="stChatMessage"] img { border-radius:12px; }
    [data-testid="stBottomBlockContainer"], [data-testid="stBottom"], [data-testid="stBottomBlockContainer"] > div {
        background:transparent !important;
        border-top:0 !important;
        box-shadow:none !important;
    }
    [data-testid="stBottomBlockContainer"] { padding:0 1rem 1.25rem !important; }
    [data-testid="stChatInput"] { max-width:820px; margin:0 auto; padding:0 !important; }
    [data-testid="stChatInput"] > div { display:flex !important; align-items:center !important; border:1px solid #41536f !important; border-radius:28px !important; background:rgba(24,35,55,.96) !important; box-shadow:0 4px 20px rgba(0,0,0,.28) !important; padding:.35rem .4rem .35rem 1rem !important; transition:border-color .2s ease, box-shadow .2s ease; }
    [data-testid="stChatInput"] > div:focus-within { border-color:#718db8 !important; box-shadow:0 0 0 2px rgba(120,183,255,.12), 0 6px 24px rgba(0,0,0,.35) !important; }
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] textarea:focus,
    [data-testid="stChatInput"] [contenteditable="true"] {
        color:#e8eef8 !important;
        -webkit-text-fill-color:#e8eef8 !important;
        caret-color:#78b7ff !important;
        border:0 !important;
        border-radius:0 !important;
        background:transparent !important;
        padding:.8rem 0 !important;
        min-height:48px !important;
        opacity:1 !important;
    }
    [data-testid="stChatInput"] textarea::placeholder { color:#a8b6ca !important; -webkit-text-fill-color:#a8b6ca !important; opacity:1 !important; }
    [data-testid="stChatInput"] button { flex:0 0 40px !important; width:40px !important; height:40px !important; color:#102039 !important; background:#78b7ff !important; border:0 !important; border-radius:50% !important; margin:0 .05rem 0 .7rem; transition:transform .2s ease, background .2s ease; }
    [data-testid="stChatInput"] button:hover { background:#9acbff !important; transform:translateY(-2px); }
    [data-testid="stTextArea"] textarea { color:var(--ink) !important; background:var(--white) !important; border:1px solid #506789 !important; border-radius:15px !important; caret-color:var(--green) !important; }
    [data-testid="stTextArea"] textarea::placeholder { color:#8da297 !important; opacity:1 !important; }
    [data-testid="stTextArea"] label, [data-testid="stNumberInput"] label, [data-testid="stSlider"] label { color:var(--ink) !important; }
    [data-testid="stTextArea"] [data-baseweb="textarea"] { background:var(--white) !important; }
    [data-testid="stNumberInput"] input { color:var(--ink) !important; background:var(--white) !important; }
    .status { color:var(--muted) !important; font-size:12px; text-align:center; margin-top:.7rem; }
    .stButton button, .stDownloadButton button { border:1px solid var(--line) !important; border-radius:9px; color:var(--ink) !important; background:rgba(24,35,55,.9) !important; font-weight:600; }
    .stButton button[kind="primary"] { color:#102039 !important; background:#78b7ff !important; border-color:#78b7ff !important; }
    .stButton button:hover, .stDownloadButton button:hover { border-color:#78b7ff !important; color:var(--green) !important; }
    div[data-baseweb="select"] > div { background:var(--white) !important; border-color:#506789 !important; color:var(--ink) !important; }
    div[data-baseweb="select"] * { color:var(--ink) !important; }
    [data-testid="stExpander"] { border-color:var(--line) !important; background:rgba(24,35,55,.55); }
    [data-testid="stExpander"] summary, [data-testid="stExpander"] label, [data-testid="stSlider"] label { color:var(--ink) !important; }
    [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { color:var(--ink); }
    [data-testid="stAlert"] { color:var(--ink) !important; }
    div[data-testid="stPopover"] button { border:0; }
    </style>
    """,
    unsafe_allow_html=True,
)


def available_models():
    try:
        result = ollama.list()
        models = result.get("models", []) if isinstance(result, dict) else getattr(result, "models", [])
        names = []
        for model in models:
            name = model.get("name") if isinstance(model, dict) else getattr(model, "model", None) or getattr(model, "name", None)
            if name:
                names.append(name)
        return names
    except Exception:
        return []


def new_chat():
    st.session_state.messages = [{"role": "system", "content": SYSTEM}]
    st.session_state.chat_started = False
    st.session_state.chat_title = "New conversation"


@st.cache_resource(show_spinner=False)
def load_image_pipeline():
    import torch
    from diffusers import AutoPipelineForText2Image

    pipeline = AutoPipelineForText2Image.from_pretrained(IMAGE_MODEL, torch_dtype=torch.float32)
    pipeline.to("cpu")
    return pipeline


def create_image(prompt, steps, guidance, seed, image_size, progress_callback):
    import torch

    pipeline = load_image_pipeline()
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def on_step_end(_pipeline, step_index, _timestep, callback_kwargs):
        progress_callback(step_index + 1, steps)
        return callback_kwargs

    return pipeline(
        prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        height=image_size,
        width=image_size,
        generator=generator,
        callback_on_step_end=on_step_end,
    ).images[0]


if "messages" not in st.session_state:
    new_chat()
if "model" not in st.session_state:
    models = available_models()
    st.session_state.model = DEFAULT_MODEL if DEFAULT_MODEL in models or not models else models[0]
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7
if "workspace_mode" not in st.session_state:
    st.session_state.workspace_mode = "Chat"

with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-mark">J</div><div><div class="brand-name">JARVIS</div><div class="brand-sub">Private AI workspace</div></div></div>', unsafe_allow_html=True)
    if st.button("+  New conversation", use_container_width=True):
        new_chat()
        st.rerun()

    st.markdown('<div class="section-label">Workspace</div>', unsafe_allow_html=True)
    st.caption("Your conversations stay in this browser session.")
    st.session_state.workspace_mode = st.radio("Workspace mode", ["Chat", "Image Studio"], key="workspace_mode_picker", horizontal=True, label_visibility="collapsed")

    if st.session_state.workspace_mode == "Image Studio":
        st.markdown('<div class="section-label">Image model</div>', unsafe_allow_html=True)
        st.caption("Hugging Face · runs locally on CPU")
        st.code(IMAGE_MODEL, language="text")
        st.markdown('<div class="status">The first image downloads the model weights.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="section-label">Model</div>', unsafe_allow_html=True)
        models = available_models()
        model_options = models or [DEFAULT_MODEL]
        selected_model = st.selectbox("Choose a local model", model_options, index=model_options.index(st.session_state.model) if st.session_state.model in model_options else 0, label_visibility="collapsed")
        st.session_state.model = selected_model

    with st.expander("Response settings"):
        st.slider("Creativity", 0.0, 1.0, st.session_state.temperature, 0.1, key="temperature")
        st.caption("Lower values are more precise. Higher values are more exploratory.")

    st.markdown('<div class="section-label">Actions</div>', unsafe_allow_html=True)
    if st.button("Clear current chat", use_container_width=True):
        new_chat()
        st.rerun()
    transcript = [message for message in st.session_state.messages if message["role"] != "system"]
    st.download_button("Export transcript", json.dumps(transcript, indent=2), "jarvis-conversation.json", "application/json", use_container_width=True, disabled=not transcript)
    st.markdown('<div class="status">Powered by Ollama and Hugging Face · Runs locally</div>', unsafe_allow_html=True)


if st.session_state.workspace_mode == "Image Studio":
    st.markdown(
        '<main class="hero"><div class="eyebrow">Local image studio</div><h1>Describe it.<br><span>Make it real.</span></h1><p>Create images privately on your computer with a lightweight Hugging Face diffusion model.</p></main>',
        unsafe_allow_html=True,
    )
    image_column, settings_column = st.columns([2.2, 1], gap="large")
    with settings_column:
        st.markdown('<div class="section-label">Generation settings</div>', unsafe_allow_html=True)
        image_size_label = st.selectbox("Output quality", ["Standard · 512 × 512", "HD · 768 × 768"])
        image_size = 512 if image_size_label.startswith("Standard") else 768
        image_steps = st.slider("Quality steps", 8, 28, 20, help="More steps improve detail but take longer on CPU.")
        image_guidance = st.slider("Prompt strength", 1.0, 12.0, 7.5, 0.5)
        image_seed = st.number_input("Seed", 0, 2147483647, 42, 1)
        st.caption("HD output and more steps improve clarity, but CPU generation can take several minutes.")
    with image_column:
        st.markdown('<div class="section-label">Prompt</div>', unsafe_allow_html=True)
        image_prompt = st.text_area("Describe your image", placeholder="A quiet Japanese garden after rain, soft morning light, editorial photography", height=140, label_visibility="collapsed")
        generate_image_button = st.button("Generate image", type="primary", use_container_width=True)
        if generate_image_button:
            if not image_prompt.strip():
                st.warning("Describe the image you want first.")
            else:
                progress = st.progress(0, text="Loading the high-quality model...")
                status = st.empty()

                def update_progress(step, total_steps):
                    progress.progress(min(step / total_steps, 1.0), text=f"Generating image · step {step} of {total_steps}")
                    status.caption(f"Rendering {image_size} × {image_size} pixels on your local CPU")

                try:
                    generated_image = create_image(image_prompt.strip(), image_steps, image_guidance, image_seed, image_size, update_progress)
                    progress.progress(1.0, text="Image ready")
                    status.empty()
                    st.image(generated_image, use_container_width=True)
                    image_bytes = BytesIO()
                    generated_image.save(image_bytes, format="PNG")
                    st.download_button("Download HD PNG", image_bytes.getvalue(), f"jarvis-{datetime.now():%Y%m%d-%H%M%S}.png", "image/png", use_container_width=True)
                except Exception as error:
                    progress.empty()
                    status.empty()
                    st.error("The local image pipeline could not start. Check the model download and try again.")
                    st.code(str(error))
    st.stop()


transcript = [message for message in st.session_state.messages if message["role"] != "system"]
starter_prompt = None
if not transcript:
    st.markdown(
        '<main class="hero"><div class="eyebrow">Your thinking partner</div><h1>Ideas in.<br><span>Clarity out.</span></h1><p>A focused, private space for writing, learning, planning, and solving problems with your local AI.</p></main>',
        unsafe_allow_html=True,
    )
    st.markdown('<div style="max-width:820px;margin:0 auto;padding:0 1rem 2rem"><div class="section-label">Start with a prompt</div></div>', unsafe_allow_html=True)
    prompt_columns = st.columns(3)
    starters = [("Shape an idea", "Turn a rough thought into a clear plan."), ("Learn something", "Explain a complex topic simply."), ("Get unstuck", "Help me think through a decision.")]
    for index, (column, (title, description)) in enumerate(zip(prompt_columns, starters)):
        with column:
            if st.button(title, key=f"starter_{index}", use_container_width=True):
                starter_prompt = {
                    "Shape an idea": "Help me turn a rough thought into a clear, practical plan.",
                    "Learn something": "Explain a complex topic simply, with an example and a short summary.",
                    "Get unstuck": "Help me think through a decision. Ask me the most useful questions first.",
                }[title]
            st.markdown(f'<div class="prompt-card"><small>{description}</small></div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="chat-wrap"><div class="chat-title"><h2>{st.session_state.chat_title}</h2><span class="status">{len(transcript)} messages</span></div></div>', unsafe_allow_html=True)
    for message in transcript:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

prompt = starter_prompt or st.chat_input("Message JARVIS...")
if prompt:
    if not st.session_state.chat_started:
        st.session_state.chat_started = True
        st.session_state.chat_title = prompt[:42] + ("..." if len(prompt) > 42 else "")
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        answer_area = st.empty()
        answer = ""
        try:
            response = ollama.chat(
                model=st.session_state.model,
                messages=st.session_state.messages,
                options={"temperature": st.session_state.temperature},
                stream=True,
            )
            for chunk in response:
                answer += chunk["message"].get("content", "")
                answer_area.markdown(answer + "▌")
            answer_area.markdown(answer)
        except Exception as error:
            answer = "I could not reach Ollama. Make sure the Ollama app is running and that the selected model is installed.\n\n`" + str(error) + "`"
            answer_area.error(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

 
 
