import os
import json
import hashlib
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import tensorflow as tf
import requests
from PIL import Image

st.set_page_config(page_title="Plant Disease AI", page_icon="🌿", layout="wide")

# ─────────────────────────────────────────────
#  API & Model  (unchanged from your original)
# ─────────────────────────────────────────────
OPENROUTER_API_KEY = "YOUR_API_KEY"

try:
    requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
        json={"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "prompt"}]},
        timeout=5
    )
except Exception:
    pass

working_dir       = os.path.dirname(os.path.abspath(__file__))
model_path        = os.path.join(working_dir, "trained_model", "plant_disease_prediction_model.h5")
class_indices_path = os.path.join(working_dir, "class_indices.json")

try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error(f"Model load error: {e}")
    model = None

try:
    with open(class_indices_path) as f:
        class_indices = json.load(f)
except Exception as e:
    st.error(f"Class indices error: {e}")
    class_indices = {}

CACHE = {}

def generate_cache_key(image):
    return hashlib.md5(image.tobytes()).hexdigest()

def load_and_preprocess_image(image, target_size=(128, 128)):
    img       = image.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array.astype("float32") / 255.0

def fetch_recommendations(disease_name):
    if disease_name in CACHE:
        return CACHE[disease_name]
    prompt = (f"The plant is healthy ({disease_name}). Give maintenance tips."
              if "healthy" in disease_name.lower()
              else f"Suggest treatment and prevention for {disease_name} in plants.")
    try:
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
            json={"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": prompt}]},
            timeout=30
        )
        data = r.json()
        if "error" in data:
            return f"API Error: {data['error']['message']}"
        result = data["choices"][0]["message"]["content"]
        CACHE[disease_name] = result
        return result
    except Exception as e:
        return f"Exception: {e}"

def predict_image_class(model, image, class_indices):
    if model is None:
        return "Model not loaded."
    ck = generate_cache_key(image)
    if ck in CACHE:
        return CACHE[ck]
    preds = model.predict(load_and_preprocess_image(image))
    if preds.size == 0:
        return "No prediction."
    idx  = int(np.argmax(preds, axis=1)[0])
    name = class_indices.get(str(idx), "Unknown class")
    CACHE[ck] = name
    return name

# ─────────────────────────────────────────────
#  Navigation via query_params
# ─────────────────────────────────────────────
PAGES = ["Home", "Demo", "Dev"]
page  = st.query_params.get("page", "Home")
if page not in PAGES:
    page = "Home"

# ─────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Syne:wght@400;600;800&display=swap');

[data-testid="stSidebar"]{display:none !important;}
[data-testid="stHeader"] {display:none !important;}
#MainMenu,footer         {display:none !important;}

html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"],.main,.block-container{
    background:#050f05 !important; color:#e8fce8 !important;
    font-family:'Syne',sans-serif !important; padding-top:0 !important;}
.block-container{padding-top:88px !important; max-width:1100px !important;}

/* ── NAVBAR ── */
.navbar{
    position:fixed;top:0;left:0;right:0;z-index:99999;
    display:flex;align-items:center;justify-content:space-between;
    padding:0 40px;height:64px;
    background:rgba(3,10,3,0.93);
    backdrop-filter:blur(24px);-webkit-backdrop-filter:blur(24px);
    border-bottom:1px solid rgba(0,255,100,0.14);
    box-shadow:0 4px 40px rgba(0,255,68,0.07);
    animation:slideDown .55s cubic-bezier(.16,1,.3,1) both;}
@keyframes slideDown{from{transform:translateY(-100%);opacity:0;}to{transform:translateY(0);opacity:1;}}
.navbar::after{
    content:'';position:absolute;top:0;left:-60%;width:55%;height:100%;
    background:linear-gradient(90deg,transparent,rgba(0,255,100,.05),transparent);
    animation:scan 4s linear infinite;pointer-events:none;}
@keyframes scan{0%{left:-60%}100%{left:120%}}

.nav-logo{
    font-family:'Orbitron',monospace;font-weight:900;font-size:1rem;
    color:#00ff99;letter-spacing:.12em;white-space:nowrap;
    text-decoration:none !important;display:flex;align-items:center;gap:9px;
    animation:logoPulse 3s ease-in-out infinite;}
@keyframes logoPulse{0%,100%{text-shadow:0 0 18px #00ff9966;}50%{text-shadow:0 0 36px #00ff99cc,0 0 60px #00ff4444;}}
.nav-dot{
    display:inline-block;width:7px;height:7px;border-radius:50%;
    background:#00ff99;box-shadow:0 0 8px #00ff99;
    animation:dotBlink 1.5s ease-in-out infinite;flex-shrink:0;}
@keyframes dotBlink{0%,100%{opacity:1;transform:scale(1);}50%{opacity:.2;transform:scale(.5);}}

.nav-links{display:flex;gap:4px;align-items:center;}
.nav-pill{
    position:relative;font-family:'Orbitron',monospace;font-size:.68rem;font-weight:700;
    letter-spacing:.1em;color:#6abf7a;padding:9px 22px;border-radius:50px;
    border:1px solid transparent;background:transparent;text-decoration:none !important;
    white-space:nowrap;overflow:hidden;
    transition:color .25s,border-color .25s,background .25s,box-shadow .25s,transform .2s;}
.nav-pill:before{
    content:'';position:absolute;inset:0;border-radius:50px;
    background:linear-gradient(90deg,#00ff9920,#00cc5510);opacity:0;transition:opacity .3s;}
.nav-pill:hover{color:#00ff99 !important;border-color:rgba(0,255,100,.4);
    box-shadow:0 0 16px rgba(0,255,100,.18);transform:translateY(-2px);}
.nav-pill:hover:before{opacity:1;}
.nav-pill.active{color:#000 !important;background:linear-gradient(90deg,#00ff99,#00dd66) !important;
    border-color:#00ff99 !important;box-shadow:0 0 22px #00ff9955,0 4px 14px #00ff4433 !important;
    transform:translateY(-2px);}
.nav-pill.active:after{
    content:'';position:absolute;bottom:0;left:50%;transform:translateX(-50%);
    width:55%;height:2px;background:#00ff99;border-radius:99px;box-shadow:0 0 8px #00ff99;}

/* ── ANIMATIONS ── */
@keyframes fadeUp{from{opacity:0;transform:translateY(28px);}to{opacity:1;transform:translateY(0);}}
@keyframes fadeLeft{from{opacity:0;transform:translateX(-30px);}to{opacity:1;transform:translateX(0);}}
@keyframes fadeRight{from{opacity:0;transform:translateX(30px);}to{opacity:1;transform:translateX(0);}}
@keyframes zoomIn{from{opacity:0;transform:scale(.93);}to{opacity:1;transform:scale(1);}}
.anim-fadeup   {animation:fadeUp    .65s cubic-bezier(.16,1,.3,1) both;}
.anim-fadeleft {animation:fadeLeft  .65s cubic-bezier(.16,1,.3,1) both;}
.anim-faderight{animation:fadeRight .65s cubic-bezier(.16,1,.3,1) both;}
.anim-zoomin   {animation:zoomIn    .65s cubic-bezier(.16,1,.3,1) both;}
.d1{animation-delay:.08s;}.d2{animation-delay:.18s;}.d3{animation-delay:.28s;}.d4{animation-delay:.38s;}

[data-testid="stMarkdownContainer"],[data-testid="stFileUploader"],
[data-testid="stImage"],[data-testid="stAlert"],div.stButton{animation:fadeUp .55s ease both;}

/* ── HEADING ── */
.main-heading{
    text-align:center;font-family:'Orbitron',monospace !important;
    font-size:clamp(1.4rem,3.5vw,2.4rem) !important;font-weight:900 !important;
    color:#00ff99 !important;text-shadow:0 0 30px #00ff9966,0 0 60px #00ff4422 !important;
    letter-spacing:.06em !important;margin-bottom:28px !important;}

/* ── GLASS CARD ── */
.glass{
    background:rgba(0,255,100,.04);border:1px solid rgba(0,255,100,.18);
    border-radius:16px;padding:28px 32px;margin-bottom:22px;
    transition:border-color .3s,box-shadow .3s,transform .3s;}
.glass:hover{border-color:rgba(0,255,100,.4);box-shadow:0 8px 40px rgba(0,255,100,.1);transform:translateY(-3px);}
.glass h2,.glass h3{font-family:'Orbitron',monospace;color:#00ff99;letter-spacing:.05em;
    margin-top:0;border-bottom:1px solid rgba(0,255,100,.2);padding-bottom:8px;}
.glass p,.glass li{color:#c8f0c8;line-height:1.8;font-family:'Syne',sans-serif;}
.glass b,.glass strong{color:#00ff99;}
.glass ol{color:#c8f0c8;padding-left:1.4rem;}
.glass ol li{margin-bottom:8px;line-height:1.8;}

/* ── MARKDOWN ── */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li{color:#c8f0c8 !important;font-family:'Syne',sans-serif !important;line-height:1.8 !important;}
[data-testid="stMarkdownContainer"] strong{color:#00ff99 !important;}
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3{font-family:'Orbitron',monospace !important;color:#00ff99 !important;
    letter-spacing:.05em !important;text-shadow:0 0 14px #00ff9944 !important;
    border-bottom:1px solid rgba(0,255,100,.2) !important;padding-bottom:8px !important;margin-top:28px !important;}
[data-testid="stMarkdownContainer"] a{color:#00ff99 !important;text-decoration:underline !important;}
[data-testid="stMarkdownContainer"] hr{border-color:rgba(0,255,100,.2) !important;margin:24px 0 !important;}
p,li{color:#c8f0c8 !important;font-family:'Syne',sans-serif !important;}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"]{border:2px dashed rgba(0,255,100,.35) !important;
    border-radius:14px !important;background:rgba(0,255,100,.04) !important;
    padding:12px !important;transition:border-color .3s,box-shadow .3s !important;}
[data-testid="stFileUploader"]:hover{border-color:#00ff99 !important;box-shadow:0 0 20px rgba(0,255,100,.1) !important;}
[data-testid="stFileUploader"] *{color:#c8f0c8 !important;font-family:'Syne',sans-serif !important;}

/* ── BUTTON ── */
div.stButton>button{
    background:linear-gradient(90deg,#00ff99,#00cc55) !important;color:#000 !important;
    font-family:'Orbitron',monospace !important;font-weight:700 !important;font-size:.82rem !important;
    letter-spacing:.06em !important;border:none !important;border-radius:8px !important;
    padding:14px 36px !important;position:relative;overflow:hidden;transition:transform .2s,box-shadow .2s !important;}
div.stButton>button:hover{transform:scale(1.05) translateY(-2px) !important;box-shadow:0 0 28px #00ff9977 !important;}

/* ── ALERTS ── */
[data-testid="stAlert"]{border-radius:12px !important;border-left:4px solid #00ff99 !important;
    background:rgba(0,255,100,.06) !important;}
[data-testid="stAlert"] *{color:#c8f0c8 !important;}

/* ── IMAGES ── */
img{border-radius:12px !important;box-shadow:0 6px 30px rgba(0,255,100,.15) !important;
    transition:transform .3s,box-shadow .3s !important;}
img:hover{transform:scale(1.01) !important;box-shadow:0 12px 40px rgba(0,255,100,.25) !important;}

[data-testid="stSpinner"] *{color:#00ff99 !important;}
::-webkit-scrollbar{width:6px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:#00ff9966;border-radius:99px;}

/* ── FOOTER ── */
.footer{position:fixed;bottom:0;left:0;width:100%;
    background:rgba(3,10,3,.93);border-top:1px solid rgba(0,255,100,.16);
    backdrop-filter:blur(12px);color:#00ff99;text-align:center;padding:10px;
    font-size:.68rem;font-family:'Orbitron',monospace;letter-spacing:.12em;z-index:99998;}

/* ── SKILL BARS ── */
.skill-wrap{margin:10px 0;}
.skill-label{font-size:.82rem;font-weight:600;color:#c8f0c8;margin-bottom:5px;
    display:flex;justify-content:space-between;}
.skill-label span{color:#00ff99;}
.skill-track{height:8px;background:rgba(0,255,100,.1);border-radius:99px;overflow:hidden;}
.skill-fill{height:100%;border-radius:99px;background:linear-gradient(90deg,#00ff99,#00cc55);
    animation:growBar 1.3s cubic-bezier(.16,1,.3,1) both;animation-delay:.3s;}
@keyframes growBar{from{width:0 !important;}}

/* ── CONTACT CHIPS ── */
.contact-row{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0;}
.contact-chip{display:flex;align-items:center;gap:8px;
    background:rgba(0,255,100,.06);border:1px solid rgba(0,255,100,.22);
    border-radius:50px;padding:9px 20px;font-size:.82rem;font-weight:600;
    color:#c8f0c8;text-decoration:none !important;
    transition:background .25s,border-color .25s,transform .2s,box-shadow .2s;}
.contact-chip:hover{background:rgba(0,255,100,.14);border-color:#00ff99;
    color:#00ff99 !important;transform:translateY(-2px);box-shadow:0 4px 18px rgba(0,255,100,.15);}

/* ── PARTICLE BG ── */
#particle-bg{position:fixed;top:0;left:0;width:100%;height:100%;
    pointer-events:none;z-index:0;opacity:.3;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Particle background
# ─────────────────────────────────────────────
st.markdown("""
<canvas id="particle-bg"></canvas>
<script>
(function(){
    var c=document.getElementById('particle-bg');
    if(!c)return;
    var x=c.getContext('2d');
    var W=c.width=window.innerWidth, H=c.height=window.innerHeight;
    window.addEventListener('resize',function(){W=c.width=window.innerWidth;H=c.height=window.innerHeight;});
    var dots=[];
    for(var i=0;i<55;i++) dots.push({
        x:Math.random()*W,y:Math.random()*H,
        vx:(Math.random()-.5)*.4,vy:(Math.random()-.5)*.4,
        r:Math.random()*1.8+.6,o:Math.random()*.6+.2});
    function draw(){
        x.clearRect(0,0,W,H);
        for(var i=0;i<dots.length;i++){
            var d=dots[i];
            d.x+=d.vx;d.y+=d.vy;
            if(d.x<0||d.x>W)d.vx*=-1;if(d.y<0||d.y>H)d.vy*=-1;
            x.beginPath();x.arc(d.x,d.y,d.r,0,Math.PI*2);
            x.fillStyle='rgba(0,255,130,'+d.o+')';x.fill();
        }
        for(var i=0;i<dots.length;i++) for(var j=i+1;j<dots.length;j++){
            var dx=dots[i].x-dots[j].x,dy=dots[i].y-dots[j].y;
            var dist=Math.sqrt(dx*dx+dy*dy);
            if(dist<120){
                x.beginPath();x.moveTo(dots[i].x,dots[i].y);x.lineTo(dots[j].x,dots[j].y);
                x.strokeStyle='rgba(0,255,100,'+(.12*(1-dist/120))+')';x.lineWidth=.6;x.stroke();
            }
        }
        requestAnimationFrame(draw);
    }
    draw();
})();
</script>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Navbar
# ─────────────────────────────────────────────
def navbar(current):
    items = [("Home","?page=Home","🏠 HOME"),("Demo","?page=Demo","🧪 DEMO"),("Dev","?page=Dev","👨‍💻 DEV")]
    pills = "".join(
        '<a class="nav-pill {a}" href="{h}">{l}</a>'.format(
            a="active" if current==k else "", h=h, l=l)
        for k,h,l in items)
    st.markdown(
        '<div class="navbar">'
        '<a class="nav-logo" href="?page=Home">🌱 PLANTAI <span class="nav-dot"></span></a>'
        '<div class="nav-links">' + pills + '</div></div>'
        '<div class="footer">© 2026 Snehal Jadhav &nbsp;|&nbsp; 🌱 Plant Disease Detection System &nbsp;|&nbsp; All Rights Reserved</div>',
        unsafe_allow_html=True)

navbar(page)

# ═══════════════════════════════════════
#  HOME
# ═══════════════════════════════════════
if page == "Home":
    st.markdown("<h1 class='main-heading anim-fadeup'>🌱 Plant Disease Detection System 🌱</h1>",
                unsafe_allow_html=True)

    st.markdown("""
    <div class="glass anim-fadeup d1">
        <h3>🌿 About Dataset</h3>
        <p>The <strong>PlantVillage dataset</strong> contains over <strong>50,000</strong> expertly curated
        images of healthy and diseased plant leaves, covering <strong>38 disease classes</strong>.
        It is the gold standard for ML-based plant disease detection research.</p>
        <h3>🔍 Project Workflow</h3>
        <ol>
            <li><strong>Collect Image Data</strong> — source from PlantVillage dataset</li>
            <li><strong>Process the Data</strong> — resize, normalize, augment</li>
            <li><strong>Split into Train / Test</strong> — 80 / 20 stratified split</li>
            <li><strong>Build the Streamlit App</strong> — this interface</li>
            <li><strong>Evaluate the Model</strong> — accuracy, loss, confusion matrix</li>
            <li><strong>CNN Training</strong> — TensorFlow / Keras deep learning model</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    workflow_image_path = (
        "C:/Users/USER/Downloads/plant-disease-prediction-cnn-deep-leanring-project-main-master/"
        "plant-disease-prediction-cnn-deep-leanring-project-main-master/app/"
        "{06441611-6856-4DD6-8748-C5EEDBCF04C3}.png"
    )
    _, mid, _ = st.columns([1, 4, 1])
    with mid:
        st.image(workflow_image_path, caption="Project Workflow Diagram", use_container_width=True)

    st.markdown("""
    <div class="glass anim-fadeup d2">
        <h3>🔍 Tech Stack</h3>
        <p>
        <strong>ML / Deep Learning:</strong> TensorFlow · Keras · Pre-trained CNN (PlantVillage)<br>
        <strong>Data Processing:</strong> Pillow (PIL) · NumPy · JSON<br>
        <strong>Web Framework:</strong> Streamlit · Custom CSS · Orbitron font<br>
        <strong>AI Recommendations:</strong> OpenRouter API (GPT-4o-mini)<br>
        <strong>System:</strong> OS Module · Hashlib Cache
        </p>
    </div>
    """, unsafe_allow_html=True)

    tech_image_path = (
        "C:/Users/USER/Downloads/plant-disease-prediction-cnn-deep-leanring-project-main-master/"
        "plant-disease-prediction-cnn-deep-leanring-project-main-master/app/Untitled design.jpg"
    )
    _, mid2, _ = st.columns([1, 4, 1])
    with mid2:
        st.image(tech_image_path, caption="Tech Stack Diagram", use_container_width=True)

# ═══════════════════════════════════════
#  DEMO
# ═══════════════════════════════════════
elif page == "Demo":
    st.markdown("<h1 class='main-heading anim-zoomin'>📷 Plant Disease Detection Demo 📷</h1>",
                unsafe_allow_html=True)

    st.markdown("""
    <div class="glass anim-fadeup d1">
        <p style="color:#c8f0c8;font-size:1rem;line-height:1.8;margin:0;">
        Upload a clear photo of a plant leaf and click <b style="color:#00ff99;">Classify Disease</b> —
        the CNN model will identify any disease and provide expert AI treatment recommendations.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        uploaded_image = st.file_uploader("📁 Upload a leaf image...", type=["jpg","jpeg","png"])
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.markdown('<p style="font-family:Orbitron,monospace;color:#00ff99;'
                    'font-size:.85rem;letter-spacing:.08em;">🔬 ANALYSIS RESULT</p>',
                    unsafe_allow_html=True)
        if uploaded_image is not None:
            if st.button("🔍 Classify Disease"):
                prediction = predict_image_class(model, image, class_indices)
                st.success(f"✅ Prediction: 🌿 {prediction} 🌿")
                with st.spinner("🌿 Getting AI recommendations..."):
                    rec = fetch_recommendations(prediction)
                st.info(f"🌱 Recommended Care:\n\n{rec}")
        else:
            st.markdown('<p style="color:rgba(0,255,100,.35);padding-top:30px;text-align:center;">'
                        '← Upload an image first</p>', unsafe_allow_html=True)

    # ──────────────────────────────────────────
    #  PIXEL PLANT DEFENDER GAME
    #  KEY FIX: Use st.components.v1.html() instead of st.markdown()
    #  This renders in a proper iframe where JavaScript actually executes!
    # ──────────────────────────────────────────
    st.markdown("""
    <div style="margin:40px 0 16px;">
        <h2 style="font-family:'Orbitron',monospace;color:#00ff99;text-align:center;
            text-shadow:0 0 20px #00ff9966;letter-spacing:.1em;">
            PLANT DEFENDER — PIXEL GAME
        </h2>
        <p style="text-align:center;color:rgba(0,255,100,.6);font-size:.78rem;
            font-family:'Orbitron',monospace;letter-spacing:.08em;">
            SHOOT THE DISEASE BUGS BEFORE THEY REACH YOUR PLANT · 3 LEVELS
        </p>
    </div>
    """, unsafe_allow_html=True)

    # The game is rendered via components.html() so JS actually runs
    components.html("""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&display=swap');
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    background:#030f03;
    display:flex; flex-direction:column; align-items:center;
    font-family:'Orbitron',monospace;
    overflow-x:hidden;
    overflow-y:auto;
    padding:0; margin:0;
  }
  #level-banner {
    font-size:.9rem; font-weight:900; color:#00ff99;
    text-shadow:0 0 20px #00ff9966;
    letter-spacing:.12em; text-align:center;
    margin:10px 0 4px;
  }
  .game-ui {
    display:flex; gap:10px; align-items:center; flex-wrap:wrap;
    justify-content:center; margin-bottom:8px;
  }
  .game-stat {
    background:rgba(0,255,100,.07); border:1px solid rgba(0,255,100,.25);
    border-radius:8px; padding:6px 14px; font-size:.65rem; color:#00ff99;
    letter-spacing:.08em;
  }
  .game-btn {
    background:linear-gradient(90deg,#00ff99,#00dd66); color:#000;
    border:none; border-radius:8px; padding:8px 18px;
    font-family:'Orbitron',monospace; font-size:.65rem; font-weight:700;
    letter-spacing:.08em; cursor:pointer;
    transition:transform .2s, box-shadow .2s;
  }
  .game-btn:hover { transform:scale(1.05); box-shadow:0 0 16px #00ff9966; }
  #game-canvas {
    border:2px solid rgba(0,255,100,.4); border-radius:8px;
    box-shadow:0 0 30px rgba(0,255,100,.15);
    image-rendering:pixelated; background:#030f03;
    display:block;
  }
  .game-controls {
    color:rgba(0,255,100,.5); font-size:.58rem;
    letter-spacing:.1em; margin-top:8px; text-align:center;
  }
  .ctrl-btn {
    background:rgba(0,255,100,.1);
    border:2px solid rgba(0,255,100,.5);
    border-radius:12px;
    color:#00ff99;
    font-family:'Orbitron',monospace;
    font-size:1.2rem;
    font-weight:900;
    padding:16px 30px;
    cursor:pointer;
    user-select:none;
    -webkit-user-select:none;
    touch-action:manipulation;
    transition:background .12s, box-shadow .12s, transform .1s;
    min-width:72px;
  }
  .ctrl-btn:active {
    background:rgba(0,255,100,.35) !important;
    box-shadow:0 0 24px #00ff9988;
    transform:scale(0.94);
  }
  .ctrl-btn:hover {
    background:rgba(0,255,100,.2);
    box-shadow:0 0 14px #00ff9944;
  }
  .shoot-btn {
    font-size:.72rem;
    padding:16px 34px;
    background:rgba(0,255,100,.18);
    border-color:#00ff99;
    box-shadow:0 0 14px #00ff9933;
    letter-spacing:.08em;
  }
</style>
</head>
<body>

<div id="level-banner">LEVEL 1 — SEEDLING</div>
<div class="game-ui">
  <div class="game-stat" id="gs-score">SCORE: 0</div>
  <div class="game-stat" id="gs-lives">LIVES: 3</div>
  <div class="game-stat" id="gs-level">LEVEL: 1/3</div>
  <div class="game-stat" id="gs-bugs">BUGS: 0/10</div>
  <button class="game-btn" onclick="startGame()">▶ START</button>
  <button class="game-btn" onclick="restartGame()">↺ RESTART</button>
</div>
<canvas id="game-canvas" width="620" height="340"></canvas>
<div class="game-controls">
  ARROW KEYS / WASD TO MOVE &nbsp;|&nbsp; SPACE OR UP TO SHOOT &nbsp;|&nbsp; OR USE BUTTONS BELOW
</div>

<!-- On-screen controls -->
<div style="display:flex;gap:12px;align-items:center;justify-content:center;margin-top:14px;flex-wrap:wrap;">
  <!-- LEFT -->
  <button class="ctrl-btn" id="btn-left"
    onpointerdown="keys['ArrowLeft']=true" onpointerup="keys['ArrowLeft']=false" onpointerleave="keys['ArrowLeft']=false">
    ◀
  </button>
  <!-- SHOOT -->
  <button class="ctrl-btn shoot-btn" id="btn-shoot"
    onpointerdown="keys[' ']=true" onpointerup="keys[' ']=false" onpointerleave="keys[' ']=false">
    🔫 SHOOT
  </button>
  <!-- RIGHT -->
  <button class="ctrl-btn" id="btn-right"
    onpointerdown="keys['ArrowRight']=true" onpointerup="keys['ArrowRight']=false" onpointerleave="keys['ArrowRight']=false">
    ▶
  </button>
</div>

<script>
(function(){
  var canvas = document.getElementById('game-canvas');
  var ctx = canvas.getContext('2d');
  var W = canvas.width, H = canvas.height;

  var LEVELS = [
    {name:'LEVEL 1 - SEEDLING', total:10, spd:0.7, frate:20, rows:2},
    {name:'LEVEL 2 - SAPLING',  total:16, spd:1.2, frate:15, rows:3},
    {name:'LEVEL 3 - FINAL BOSS',total:24,spd:1.8, frate:10, rows:4}
  ];

  var state='idle', score=0, lives=3, lvl=0, kills=0;
  var player, bullets, bugs, particles, bgStars;
  var bugDir=1, frame=0, cooldown=0;
  var keys={}; window.keys=keys;

  function hud(){
    var cfg=LEVELS[lvl];
    document.getElementById('gs-score').textContent='SCORE: '+score;
    document.getElementById('gs-lives').textContent='LIVES: '+lives;
    document.getElementById('gs-level').textContent='LEVEL: '+(lvl+1)+'/3';
    document.getElementById('gs-bugs').textContent='BUGS: '+kills+'/'+cfg.total;
    document.getElementById('level-banner').textContent=cfg.name;
  }

  function initGame(){
    var cfg=LEVELS[lvl];
    player={x:W/2-12, y:H-50, w:24, h:24, spd:3.5};
    bullets=[]; particles=[];
    bugDir=1; frame=0; cooldown=0; kills=0;

    bugs=[];
    var cols=Math.ceil(cfg.total/cfg.rows);
    var count=0;
    for(var r=0;r<cfg.rows&&count<cfg.total;r++){
      for(var c=0;c<cols&&count<cfg.total;c++){
        bugs.push({
          x:50+c*50, y:40+r*34,
          w:18, h:14, alive:true,
          at:Math.random()*6.28,
          row:r,
          ebullets:[],
          stimer:60+Math.floor(Math.random()*80)
        });
        count++;
      }
    }
    bgStars=[];
    for(var i=0;i<40;i++) bgStars.push({
      x:Math.random()*W, y:Math.random()*H,
      r:Math.random()*1.2+.3,
      o:Math.random()*.4+.1,
      vy:Math.random()*.3+.1
    });
    hud();
  }

  function pr(x,y,w,h,col,glow){
    if(glow){ctx.shadowBlur=10;ctx.shadowColor=col;}
    ctx.fillStyle=col;
    ctx.fillRect(Math.round(x),Math.round(y),w,h);
    ctx.shadowBlur=0;
  }

  function drawPlayer(p){
    pr(p.x+8,  p.y,    8, 6,  '#00ff99',true);
    pr(p.x+4,  p.y+6,  16,8,  '#00cc55',true);
    pr(p.x,    p.y+8,  24,12, '#00ff99',true);
    pr(p.x+8,  p.y+14, 8, 8,  '#00cc55',false);
    pr(p.x+10, p.y-4,  4, 6,  '#aaff44',true);
    ctx.fillStyle='rgba(0,255,100,.4)';
    ctx.beginPath();
    ctx.moveTo(p.x,p.y+10);ctx.lineTo(p.x-10,p.y+20);ctx.lineTo(p.x,p.y+22);
    ctx.moveTo(p.x+p.w,p.y+10);ctx.lineTo(p.x+p.w+10,p.y+20);ctx.lineTo(p.x+p.w,p.y+22);
    ctx.fill();
  }

  var BUGCOLS=['#ff4444','#ff8800','#ff00cc'];
  function drawBug(b){
    if(!b.alive)return;
    var col=BUGCOLS[b.row%3];
    var blink=Math.sin(b.at+frame*.1)>0;
    ctx.shadowBlur=8;ctx.shadowColor=col;
    ctx.fillStyle=col;
    ctx.fillRect(Math.round(b.x+4),Math.round(b.y),10,7);
    ctx.fillRect(Math.round(b.x+2),Math.round(b.y+4),14,7);
    ctx.fillRect(Math.round(b.x+4),Math.round(b.y+8),10,4);
    ctx.shadowBlur=0;
    ctx.fillStyle=blink?'#fff':col;
    ctx.fillRect(Math.round(b.x+5),Math.round(b.y+1),3,3);
    ctx.fillRect(Math.round(b.x+10),Math.round(b.y+1),3,3);
    ctx.strokeStyle=col;ctx.lineWidth=1;
    ctx.beginPath();
    ctx.moveTo(b.x+6,b.y);ctx.lineTo(b.x+3,b.y-5);
    ctx.moveTo(b.x+12,b.y);ctx.lineTo(b.x+15,b.y-5);
    ctx.stroke();
  }

  function drawBullet(b){
    ctx.shadowBlur=8;ctx.shadowColor='#aaff44';
    ctx.fillStyle='#aaff44';
    ctx.fillRect(Math.round(b.x),Math.round(b.y),3,9);
    ctx.shadowBlur=0;
  }

  function drawEBullet(b){
    ctx.shadowBlur=6;ctx.shadowColor='#ff4444';
    ctx.fillStyle='#ff4444';
    ctx.fillRect(Math.round(b.x),Math.round(b.y),3,7);
    ctx.shadowBlur=0;
  }

  function spawnFX(x,y,col,n){
    for(var i=0;i<n;i++) particles.push({
      x:x,y:y,
      vx:(Math.random()-.5)*4,
      vy:(Math.random()-.5)*4-1,
      s:Math.floor(Math.random()*4)+2,
      col:col,life:30,max:30
    });
  }

  function drawFX(){
    for(var i=particles.length-1;i>=0;i--){
      var p=particles[i];
      ctx.globalAlpha=p.life/p.max;
      ctx.fillStyle=p.col;
      ctx.fillRect(Math.round(p.x),Math.round(p.y),p.s,p.s);
      p.x+=p.vx;p.y+=p.vy;p.vy+=.08;p.life--;
      if(p.life<=0)particles.splice(i,1);
    }
    ctx.globalAlpha=1;
  }

  function drawBG(){
    for(var i=0;i<bgStars.length;i++){
      var s=bgStars[i];
      s.y+=s.vy;if(s.y>H)s.y=0;
      ctx.globalAlpha=s.o;
      ctx.fillStyle='rgba(0,255,100,.6)';
      ctx.fillRect(Math.round(s.x),Math.round(s.y),Math.round(s.r*2),Math.round(s.r*2));
    }
    ctx.globalAlpha=1;
    ctx.fillStyle='rgba(0,255,100,.12)';
    ctx.fillRect(0,H-14,W,3);
    ctx.strokeStyle='rgba(0,255,100,.03)';ctx.lineWidth=1;
    for(var gx=0;gx<W;gx+=64){ctx.beginPath();ctx.moveTo(gx,0);ctx.lineTo(gx,H);ctx.stroke();}
    for(var gy=0;gy<H;gy+=64){ctx.beginPath();ctx.moveTo(0,gy);ctx.lineTo(W,gy);ctx.stroke();}
  }

  function overlay(t1,t2){
    ctx.fillStyle='rgba(0,0,0,.65)';ctx.fillRect(0,0,W,H);
    ctx.shadowBlur=24;ctx.shadowColor='#00ff99';
    ctx.fillStyle='#00ff99';
    ctx.font='bold 26px Orbitron,monospace';
    ctx.textAlign='center';
    ctx.fillText(t1,W/2,H/2-16);
    ctx.shadowBlur=0;
    ctx.fillStyle='rgba(0,255,100,.7)';
    ctx.font='13px Orbitron,monospace';
    ctx.fillText(t2,W/2,H/2+18);
    ctx.textAlign='left';
  }

  function loop(){
    requestAnimationFrame(loop);
    ctx.fillStyle='#030f03';ctx.fillRect(0,0,W,H);
    drawBG();

    if(state==='idle'){overlay('PLANT DEFENDER','CLICK START TO PLAY');return;}
    if(state==='dead'){overlay('GAME OVER','SCORE: '+score+' | CLICK RESTART');return;}
    if(state==='win') {overlay('YOU WIN!','FINAL SCORE: '+score+' — AMAZING!');return;}
    if(state==='lvlup'){overlay('LEVEL UP!','GET READY...');return;}

    frame++;
    cooldown=Math.max(0,cooldown-1);
    var cfg=LEVELS[lvl];

    // move player
    if(keys['ArrowLeft']||keys['a']||keys['A']) player.x=Math.max(0,player.x-player.spd);
    if(keys['ArrowRight']||keys['d']||keys['D']) player.x=Math.min(W-player.w,player.x+player.spd);

    // shoot
    if((keys['ArrowUp']||keys[' ']||keys['w']||keys['W'])&&cooldown===0){
      bullets.push({x:player.x+11,y:player.y-4,spd:7});
      cooldown=cfg.frate;
    }

    // move player bullets
    for(var i=bullets.length-1;i>=0;i--){
      bullets[i].y-=bullets[i].spd;
      if(bullets[i].y<-10)bullets.splice(i,1);
    }

    // move bugs
    var alive=bugs.filter(function(b){return b.alive;});
    var edge=false;
    for(var i=0;i<alive.length;i++){
      alive[i].x+=bugDir*cfg.spd;
      if(alive[i].x+alive[i].w>=W-18||alive[i].x<=18)edge=true;
    }
    if(edge){
      bugDir*=-1;
      for(var i=0;i<alive.length;i++) alive[i].y+=12;
    }

    // bug shooting logic
    for(var i=0;i<alive.length;i++){
      var b=alive[i];
      b.stimer--;
      if(b.stimer<=0){
        b.ebullets.push({x:b.x+7,y:b.y+b.h,spd:2+lvl*.5});
        b.stimer=70+Math.floor(Math.random()*70)-lvl*8;
      }
      for(var j=b.ebullets.length-1;j>=0;j--){
        b.ebullets[j].y+=b.ebullets[j].spd;
        if(b.ebullets[j].y>H){b.ebullets.splice(j,1);continue;}
        var eb=b.ebullets[j];
        if(eb.x>player.x&&eb.x<player.x+player.w&&eb.y>player.y&&eb.y<player.y+player.h){
          spawnFX(player.x+12,player.y+10,'#ff3333',12);
          b.ebullets.splice(j,1);
          lives--;hud();
          if(lives<=0)state='dead';
        }
      }
      if(b.y+b.h>=H-18)state='dead';
    }

    // player bullet vs bug
    outer:
    for(var bi=bullets.length-1;bi>=0;bi--){
      for(var gi=0;gi<bugs.length;gi++){
        var g=bugs[gi];
        if(!g.alive)continue;
        var bul=bullets[bi];
        if(bul&&bul.x>g.x&&bul.x<g.x+g.w&&bul.y>g.y&&bul.y<g.y+g.h){
          g.alive=false;
          bullets.splice(bi,1);
          score+=(lvl+1)*10;kills++;
          spawnFX(g.x+9,g.y+7,BUGCOLS[g.row%3],10);
          hud();
          continue outer;
        }
      }
    }

    // check level done
    if(kills>=cfg.total){
      if(lvl<2){
        state='lvlup';
        setTimeout(function(){lvl++;initGame();state='playing';hud();},2000);
      }else{state='win';}
    }

    // draw everything
    for(var i=0;i<bugs.length;i++){
      drawBug(bugs[i]);
      for(var j=0;j<bugs[i].ebullets.length;j++) drawEBullet(bugs[i].ebullets[j]);
    }
    for(var i=0;i<bullets.length;i++) drawBullet(bullets[i]);
    drawFX();
    drawPlayer(player);

    // HUD bar at bottom
    ctx.fillStyle='rgba(0,255,100,.08)';ctx.fillRect(0,H-14,W,14);
    ctx.fillStyle='#00ff99';ctx.font='9px Orbitron,monospace';
    ctx.fillText('SCORE:'+score+'  LIVES:'+lives+'  LEVEL:'+(lvl+1),10,H-3);
  }

  // Key listeners — attached to the document inside the iframe
  document.addEventListener('keydown',function(e){
    keys[e.key]=true;
    if(e.key===' ')e.preventDefault();
  });
  document.addEventListener('keyup',function(e){keys[e.key]=false;});

  // Also support clicking canvas to focus it for key events
  canvas.addEventListener('click', function(){ canvas.focus(); });

  window.startGame=function(){score=0;lives=3;lvl=0;kills=0;initGame();state='playing';hud();};
  window.restartGame=function(){score=0;lives=3;lvl=0;kills=0;initGame();state='playing';hud();};

  initGame();
  loop();
})();
</script>
</body>
</html>
""", height=560, scrolling=False)

# ═══════════════════════════════════════
#  DEV
# ═══════════════════════════════════════
elif page == "Dev":
    st.markdown("<h1 class='main-heading anim-zoomin'>🚀 Developer Information 🚀</h1>",
                unsafe_allow_html=True)

    dev_img = (
        "C:/Users/USER/Downloads/plant-disease-prediction-cnn-deep-leanring-project-main-master/"
        "plant-disease-prediction-cnn-deep-leanring-project-main-master/app/"
        "WhatsApp Image 2026-01-12 at 15.06.31.jpeg"
    )
    _, mid, _ = st.columns([1, 3, 1])
    with mid:
        st.image(dev_img, caption="Snehal Jadhav", use_container_width=True)

    st.markdown("""
    <div class="glass anim-fadeup d1">
        <h2>📬 Connect with Me</h2>
        <div class="contact-row">
            <a class="contact-chip" href="mailto:buisnesssnehal@example.com">📧 buisnesssnehal@example.com</a>
            <a class="contact-chip" href="https://github.com/snehal395u" target="_blank">🐙 GitHub</a>
            <a class="contact-chip" href="https://www.linkedin.com/in/snehal-jadhav-1b1101305/" target="_blank">💼 LinkedIn</a>
        </div>
        <p style="text-align:center;margin-top:16px;color:#c8f0c8;">
            <b>Feel free to reach out! Let's build something amazing together. 🚀</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("""
        <div class="glass anim-fadeleft d2">
            <h3>👋 About Me</h3>
            <p>A passionate <b>Software Developer, AI Engineer</b> and
            <b>Data Science Enthusiast</b>, currently in my <b>third year</b>
            of BSc Computer Science.<br><br>
            I love working with <b>Deep Learning, Backend Development</b> and <b>DSA</b>.</p>
        </div>
        <div class="glass anim-fadeleft d3">
            <h3>🛠️ Tech Stack</h3>
            <p>
            <b>ML/DL:</b> TensorFlow · Keras · PyTorch<br>
            <b>Backend:</b> Java · Spring Boot · WebFlux · Hibernate · Microservices<br>
            <b>Frontend:</b> React · Next.js · Streamlit · Flask · Django<br>
            <b>Database:</b> MongoDB · MySQL · Oracle · Firebase<br>
            <b>DSA:</b> Java · Python · Design Patterns
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        skills = [
            ("Python & ML",       92),
            ("TensorFlow / Keras", 86),
            ("Java & Spring Boot", 82),
            ("React / Next.js",    78),
            ("Streamlit & UI",     90),
            ("Database Design",    80),
        ]
        bars = ""
        for label, pct in skills:
            bars += (
                '<div class="skill-wrap">'
                '<div class="skill-label">' + label +
                '<span>' + str(pct) + '%</span></div>'
                '<div class="skill-track">'
                '<div class="skill-fill" style="width:' + str(pct) + '%;"></div>'
                '</div></div>'
            )

        st.markdown(
            '<div class="glass anim-faderight d2"><h3>📈 Skill Levels</h3>' + bars + '</div>',
            unsafe_allow_html=True)

        st.markdown("""
        <div class="glass anim-faderight d3">
            <h3>🌱 This Project</h3>
            <p>A <b>CNN-based Plant Disease Detection</b> system trained on
            the PlantVillage dataset (50,000+ images, 38 classes), deployed
            with Streamlit and powered by <b>OpenRouter AI</b> for real-time
            treatment recommendations.</p>
        </div>
        """, unsafe_allow_html=True)
