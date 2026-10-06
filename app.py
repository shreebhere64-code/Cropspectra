from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import sqlite3
import bcrypt
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import json
import csv
from gtts import gTTS
from deep_translator import GoogleTranslator

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "supersecretkey")
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.context_processor
def inject_user():
    return dict(user=session.get("user"))


# ---------------- DATABASE SETUP ----------------
DB_NAME = os.path.join(app.root_path, "users.db")


def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            username TEXT UNIQUE,
            email TEXT UNIQUE,
            password TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


init_db()

# ---------------- LOAD ML MODELS & DATA ----------------
print("[INFO] Loading ML Models...")

# 1. Solanaceae Model (Tomato, Potato, Bell Pepper)
main_model_path = os.path.join(app.root_path, "crop_disease_model.h5")
model_solanaceae = tf.keras.models.load_model(main_model_path)

solanaceae_classes = [
    ".ipynb_checkpoints",
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "PlantVillage",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

# 2. Maize (Corn) Model
maize_model_dir = os.path.join(app.root_path, "maize_model")
model_maize = tf.saved_model.load(maize_model_dir)
maize_infer = model_maize.signatures["serving_default"]
maize_in_key = list(maize_infer.structured_input_signature[1].keys())[0]

maize_classes = [
    "Corn_(maize)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___Common_rust_"
]

# 3. Rice Model
rice_model_path = os.path.join(app.root_path, "rice_disease_model.h5")
model_rice = tf.keras.models.load_model(rice_model_path)

rice_classes = [
    "Rice___healthy",
    "Rice___Bacterial_leaf_blight",
    "Rice___Brown_spot",
    "Rice___Leaf_Blast"
]

# 4. Wheat Model
wheat_model_path = os.path.join(app.root_path, "wheat_disease_model.h5")
model_wheat = tf.keras.models.load_model(wheat_model_path)

wheat_classes = [
    "Wheat___Brown_rust",
    "Wheat___healthy",
    "Wheat___Loose_smut",
    "Wheat___Septoria",
    "Wheat___Yellow_rust"
]

# 5. Feature Extractor & ImageNet Model for OOD validation
feat_extractor = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
imgnet_model = MobileNetV2(weights="imagenet", include_top=True)

# Load plant leaf reference prototype vector
proto_path = os.path.join(app.root_path, "plant_prototype.npy")
if os.path.exists(proto_path):
    ref_proto = np.load(proto_path)
else:
    sample_img_path = os.path.join(app.root_path, "static", "fc5c5672-d1e5-4374-bd99-608d95609a7f___RS_HL 0474.JPG")
    if os.path.exists(sample_img_path):
        sample_img = image.load_img(sample_img_path, target_size=(224, 224))
        f_sample = feat_extractor.predict(preprocess_input(np.expand_dims(image.img_to_array(sample_img), axis=0)), verbose=0)[0]
        ref_proto = f_sample / (np.linalg.norm(f_sample) + 1e-7)
        np.save(proto_path, ref_proto)
    else:
        ref_proto = np.ones((1280,), dtype=np.float32) / np.sqrt(1280)

# Load disease advisory knowledge base
disease_info_path = os.path.join(app.root_path, "disease_info.json")
with open(disease_info_path, "r", encoding="utf-8") as f:
    disease_info = json.load(f)

classes_path = os.path.join(app.root_path, "classes.json")
with open(classes_path, "r", encoding="utf-8") as f:
    class_names = json.load(f)

print("[INFO] All 4 Crop Disease Models & Knowledge Base loaded successfully!")

# Keywords that indicate botanical / plant content in ImageNet classifications
PLANT_KEYWORDS = {
    'plant', 'leaf', 'flower', 'tree', 'crop', 'vegetable', 'fruit', 'cabbage', 'corn', 'ear',
    'pot', 'vase', 'gardening', 'zucchini', 'cucumber', 'artichoke', 'bell_pepper', 'pepper',
    'broccoli', 'cauliflower', 'head_cabbage', 'grass', 'lawn', 'hay', 'straw', 'wheat', 'grain',
    'ear_of_corn', 'acorn_squash', 'butternut_squash', 'daisy', 'yellow_lady_slipper', 'sunflower',
    'fungus', 'mushroom', 'lichen', 'spore', 'moss', 'botany', 'foliage', 'herb', 'flora'
}


def check_is_plant(img_array_224):
    """
    Checks if the image represents a plant/crop using ImageNet classifications & color variance.
    Returns (is_plant: bool, reason: str, confidence: float)
    """
    # Color variance check (detect solid colors, blank images)
    std_dev = float(np.std(img_array_224))
    if std_dev < 18.0:
        return False, "Low image variance / blank image", 0.0

    # ImageNet top-5 classification
    arr_pre = preprocess_input(np.expand_dims(img_array_224.copy(), axis=0))
    preds = decode_predictions(imgnet_model.predict(arr_pre, verbose=0), top=5)[0]

    top_label = preds[0][1].lower()
    top_score = float(preds[0][2])

    has_plant_signal = any(
        any(k in p[1].lower() for k in PLANT_KEYWORDS)
        for p in preds[:3]
    )

    # If strongly non-plant (confidence > 0.40) and zero plant keywords in top 3
    if top_score > 0.40 and not has_plant_signal:
        return False, f"Detected non-plant object: {top_label}", top_score

    return True, top_label, top_score


def predict_crop_disease(filepath):
    """
    Runs multi-crop inference across Solanaceae, Maize, Rice, and Wheat models.
    Validates crop features and applies cosine similarity thresholding for Out-of-Distribution detection.
    """
    # Load original image for multi-size scaling
    img_pil = image.load_img(filepath)

    # 224x224 for Solanaceae, Maize, and MobileNetV2
    img224 = img_pil.resize((224, 224))
    arr224 = image.img_to_array(img224)

    # 1. Out-of-Distribution / Non-Plant Check
    is_plant, reason, _ = check_is_plant(arr224)
    if not is_plant:
        return "Disease not found", 0.0

    # 2. Feature Prototype Cosine Similarity
    f = feat_extractor.predict(preprocess_input(np.expand_dims(arr224, axis=0)), verbose=0)[0]
    f_norm = f / (np.linalg.norm(f) + 1e-7)
    sim = float(np.dot(f_norm, ref_proto))
    if sim < 0.55:
        return "Disease not found", sim

    # 3. Model Inferences
    # A. Solanaceae Model (Tomato, Potato, Pepper)
    arr224_norm = np.expand_dims(arr224 / 255.0, axis=0)
    pred_sol = model_solanaceae.predict(arr224_norm, verbose=0)[0]
    valid_sol_indices = [i for i, c in enumerate(solanaceae_classes) if not c.startswith(".") and c != "PlantVillage"]
    best_sol_idx = max(valid_sol_indices, key=lambda i: pred_sol[i])
    best_sol_conf = float(pred_sol[best_sol_idx])
    best_sol_class = solanaceae_classes[best_sol_idx]

    # B. Maize Model
    arr_maize = tf.constant(np.expand_dims(arr224, axis=0), dtype=tf.float32)
    pred_maize = maize_infer(**{maize_in_key: arr_maize})["output_0"].numpy()[0]
    best_maize_idx = int(np.argmax(pred_maize))
    best_maize_conf = float(pred_maize[best_maize_idx])
    best_maize_class = maize_classes[best_maize_idx]

    # C. Wheat Model (128x128)
    img128 = img_pil.resize((128, 128))
    arr128_norm = np.expand_dims(image.img_to_array(img128) / 255.0, axis=0)
    pred_wheat = model_wheat.predict(arr128_norm, verbose=0)[0]
    best_wheat_idx = int(np.argmax(pred_wheat))
    best_wheat_conf = float(pred_wheat[best_wheat_idx])
    best_wheat_class = wheat_classes[best_wheat_idx]

    # D. Rice Model (64x64)
    img64 = img_pil.resize((64, 64))
    arr64_norm = np.expand_dims(image.img_to_array(img64) / 255.0, axis=0)
    pred_rice = model_rice.predict(arr64_norm, verbose=0)[0]
    best_rice_idx = int(np.argmax(pred_rice))
    best_rice_conf = float(pred_rice[best_rice_idx])
    best_rice_class = rice_classes[best_rice_idx]

    # 4. Hierarchical Decision Routing
    # If Solanaceae model is very confident (Tomato, Potato, Pepper)
    if best_sol_conf >= 0.65:
        return best_sol_class, best_sol_conf

    # If Maize model detects a disease with good confidence or healthy
    if (best_maize_class != "Corn_(maize)___healthy" and best_maize_conf >= 0.65) or best_maize_conf >= 0.85:
        return best_maize_class, best_maize_conf

    # If Wheat model detects a disease with good confidence
    if best_wheat_conf >= 0.55:
        return best_wheat_class, best_wheat_conf

    # If Rice model detects a disease with good confidence
    if best_rice_conf >= 0.70:
        return best_rice_class, best_rice_conf

    # Overall candidate ranking fallback
    candidates = [
        (best_sol_class, best_sol_conf),
        (best_maize_class, best_maize_conf),
        (best_wheat_class, best_wheat_conf),
        (best_rice_class, best_rice_conf)
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)
    top_class, top_conf = candidates[0]

    if top_conf < 0.58:
        return "Disease not found", top_conf

    return top_class, top_conf


# ---------------- HOME PAGE ----------------
@app.route("/")
def index():
    return redirect(url_for("home"))


@app.route("/home")
def home():
    return render_template("home.html", user=session.get("user"))


# ---------------- SIGNUP ----------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if password != confirm_password:
            flash("Passwords do not match!", "error")
            return redirect(url_for("signup"))

        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()

        # Check if username or email exists
        cursor.execute("SELECT * FROM users WHERE username=? OR email=?", (username, email))
        existing_user = cursor.fetchone()
        if existing_user:
            flash("Username or Email already registered!", "error")
            conn.close()
            return redirect(url_for("signup"))

        # Hash password
        hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        cursor.execute("INSERT INTO users (name, username, email, password) VALUES (?, ?, ?, ?)",
                       (name, username, email, hashed_pw))
        conn.commit()
        conn.close()

        flash("Signup successful! Please login.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")


# ---------------- LOGIN ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username_email = request.form.get("username_email", "").strip()
        password = request.form.get("password", "")

        # Direct quick login for demo / judging
        if username_email.lower() == "crop" and password == "crop":
            session.clear()
            session["user"] = "crop"
            flash("Welcome crop! (Direct Access)", "success")
            return redirect(url_for("home"))

        # Database login
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT username, email, password FROM users WHERE username=? OR email=?", (username_email, username_email))
        user = cursor.fetchone()
        conn.close()

        if user:
            stored_pw = user[2].encode('utf-8')
            if bcrypt.checkpw(password.encode('utf-8'), stored_pw):
                session.clear()
                session["user"] = user[0]
                flash(f"Welcome {user[0]}!", "success")
                return redirect(url_for("home"))
            else:
                flash("Invalid username/email or password!", "error")
                return redirect(url_for("login"))
        else:
            flash("User not found! Please signup.", "error")
            return redirect(url_for("signup"))

    return render_template("login.html", user=None)


# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("Logged out successfully!", "success")
    return redirect(url_for("login"))


# ---------------- FORGOT PASSWORD ----------------
@app.route("/forgot_password")
def forgot_password():
    flash("Password reset functionality is not implemented yet.", "error")
    return redirect(url_for("login"))


# ---------------- PREDICT PAGE ----------------
@app.route("/predict", methods=["GET", "POST"])
def predict_page():
    if "user" not in session:
        flash("Please login first!", "error")
        return redirect(url_for("login"))

    if request.method == "POST":
        if 'file' not in request.files:
            flash("No file uploaded!", "error")
            return redirect(url_for("predict_page"))

        file = request.files["file"]
        if file.filename == "":
            flash("No file selected!", "error")
            return redirect(url_for("predict_page"))

        # Save uploaded file
        filename = file.filename
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Predict disease using Multi-Crop & OOD Engine
        predicted_raw, confidence = predict_crop_disease(filepath)

        if predicted_raw in ["Disease not found", "Not a valid crop image"]:
            predicted_display = "Disease Not Found"
            info = disease_info.get("Disease not found", {})
            description = info.get("description", "The uploaded image does not match any recognized disease from our trained crop classes (Tomato, Potato, Bell Pepper, Maize, Rice, Wheat).")
            symptoms = []
            treatment = "No treatment available because no disease was detected."
            prevention = "Ensure you capture a clear, well-lit photograph focusing directly on the infected crop leaf."
        else:
            info = disease_info.get(predicted_raw, {})
            crop = info.get("crop", "")
            d_name = info.get("disease_name", "")
            if crop and d_name:
                predicted_display = f"{crop} - {d_name}"
            else:
                predicted_display = predicted_raw.replace("___", " - ").replace("__", " - ").replace("_", " ")

            description = info.get("description", "No description available.")
            raw_symptoms = info.get("symptoms", [])
            symptoms = raw_symptoms if isinstance(raw_symptoms, list) else [raw_symptoms]

            raw_treatment = info.get("treatment", "No treatment details available.")
            treatment = " ".join(raw_treatment) if isinstance(raw_treatment, list) else raw_treatment

            raw_prevention = info.get("prevention", "No prevention details available.")
            prevention = " ".join(raw_prevention) if isinstance(raw_prevention, list) else raw_prevention

        web_img_path = url_for('static', filename=f'uploads/{filename}')

        return render_template(
            "predict.html",
            prediction=predicted_display,
            raw_prediction=predicted_raw,
            confidence=f"{confidence*100:.1f}%" if confidence > 0 else "0%",
            description=description,
            symptoms=symptoms,
            treatment=treatment,
            prevention=prevention,
            img_path=web_img_path
        )

    return render_template("predict.html")


# ---------------- TEXT TO SPEECH ----------------
@app.route("/text_to_speech", methods=["POST"])
def text_to_speech():
    if "user" not in session:
        return {"error": "Unauthorized"}, 401

    data = request.get_json() or {}
    prediction = data.get("prediction", "")
    description = data.get("description", "")
    symptoms = data.get("symptoms", [])
    treatment = data.get("treatment", "")
    prevention = data.get("prevention", "")

    # Build clear spoken text
    text_to_speak = f"Prediction: {prediction}. "
    text_to_speak += f"Description: {description}. "

    if symptoms and len(symptoms) > 0:
        text_to_speak += "Symptoms: " + ", ".join(symptoms) + ". "

    if treatment and treatment != "N/A":
        text_to_speak += f"Treatment: {treatment}. "
    if prevention and prevention != "N/A":
        text_to_speak += f"Prevention: {prevention}."

    audio_filename = "prediction_speech.mp3"
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)

    try:
        tts = gTTS(text=text_to_speak, lang='en', slow=False)
        tts.save(audio_path)
        return {"success": True, "audio_url": url_for('static', filename=f'uploads/{audio_filename}')}
    except Exception as e:
        print(f"TTS Error: {str(e)}")
        return {"error": str(e)}, 500


# ---------------- TRANSLATE TEXT ----------------
@app.route("/translate_text", methods=["POST"])
def translate_text():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json() or {}
    target_lang = data.get("target_lang", "hi")
    prediction = data.get("prediction", "")
    description = data.get("description", "")
    symptoms = data.get("symptoms", [])
    treatment = data.get("treatment", "")
    prevention = data.get("prevention", "")

    trans_lang = "hi" if target_lang == "bho" else target_lang

    try:
        translator = GoogleTranslator(source='auto', target=trans_lang)
        translated_prediction = translator.translate(prediction) if prediction else ""
        translated_description = translator.translate(description) if description else ""

        translated_symptoms = []
        if symptoms:
            for sym in symptoms:
                try:
                    translated_symptoms.append(translator.translate(sym))
                except Exception:
                    translated_symptoms.append(sym)

        translated_treatment = translator.translate(treatment) if treatment else ""
        translated_prevention = translator.translate(prevention) if prevention else ""

        return jsonify({
            "success": True,
            "translated": {
                "prediction": translated_prediction,
                "description": translated_description,
                "symptoms": translated_symptoms,
                "treatment": translated_treatment,
                "prevention": translated_prevention
            }
        })
    except Exception as e:
        print(f"Translation Error: {str(e)}")
        return jsonify({"error": "Translation service temporarily unavailable. Please try again."}), 500


# ---------------- TRANSLATED TEXT TO SPEECH ----------------
@app.route("/translated_speech", methods=["POST"])
def translated_speech():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json() or {}
    target_lang = data.get("target_lang", "hi")
    prediction = data.get("prediction", "")
    description = data.get("description", "")
    symptoms = data.get("symptoms", [])
    treatment = data.get("treatment", "")
    prevention = data.get("prevention", "")

    text_to_speak = f"{prediction}. {description}. "

    if symptoms and len(symptoms) > 0:
        text_to_speak += " ".join(symptoms) + ". "

    if treatment:
        text_to_speak += f"{treatment}. "
    if prevention:
        text_to_speak += f"{prevention}."

    lang_map = {
        "hi": "hi",
        "bn": "bn",
        "mr": "mr",
        "bho": "hi"
    }

    speech_lang = lang_map.get(target_lang, "hi")
    audio_filename = f"translated_speech_{target_lang}.mp3"
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)

    try:
        tts = gTTS(text=text_to_speak, lang=speech_lang, slow=False)
        tts.save(audio_path)
        return jsonify({"success": True, "audio_url": url_for('static', filename=f'uploads/{audio_filename}')})
    except Exception as e:
        print(f"TTS Error: {str(e)}")
        return jsonify({"error": "Speech generation failed. Please try again."}), 500


# ---------------- EXTRA PAGES ----------------
@app.route("/about")
def about():
    return render_template("about.html", user=session.get("user"))


@app.route("/blogs")
def blogs():
    return render_template("blogs.html", user=session.get("user"))


@app.route("/contact")
def contact():
    return render_template("contact.html", user=session.get("user"))


# ---------------- FEEDBACK SUBMIT ----------------
@app.route("/submit_feedback", methods=["POST"])
def submit_feedback():
    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip()
    subject = request.form.get("subject", "").strip()
    message = request.form.get("message", "").strip()

    feedback_path = os.path.join(app.root_path, "feedback.csv")
    with open(feedback_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([name, email, subject, message])

    flash("Your message has been sent successfully! ✅", "success")
    return redirect(url_for("contact"))


# ---------------- RUN APP ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
