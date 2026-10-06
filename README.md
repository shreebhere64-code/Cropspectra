# 🌾 CropSpectra - AI-Powered Crop Disease Detection System

**CropSpectra** is an intelligent crop disease prediction system that uses **Deep Learning** and **Computer Vision** to detect diseases in crops like Tomato, Potato, and Bell Pepper.

---

## 🚀 Features

✅ **AI-Based Disease Detection** - Deep learning powered predictions  
✅ **Multi-Language Support** - Hindi, Bengali, Marathi, Bhojpuri  
✅ **Text-to-Speech** - Audio output in multiple languages  
✅ **User Authentication** - Secure login system  
✅ **Real-Time Prediction** - Instant disease identification  

---

## 📋 System Requirements

- **Python**: 3.8, 3.9, or 3.10 (Recommended: 3.9)
- **Operating System**: Windows / Linux / macOS
- **RAM**: Minimum 4GB (8GB recommended)
- **Storage**: At least 2GB free space

---

## 📦 Quick Start Guide

### **Windows Users:**

```cmd
# 1. Extract the project folder
# 2. Open Command Prompt in project directory
cd c:\Users\abhin\Downloads\Abhi\Abhi

# 3. Activate virtual environment
venv\Scripts\activate

# 4. Run the application
python app.py
```

### **Linux/macOS Users:**

```bash
# 1. Extract the project folder
# 2. Open Terminal in project directory
cd /path/to/Abhi/Abhi

# 3. Activate virtual environment
source venv/bin/activate

# 4. Run the application
python app.py
```

✅ **Success!** Open browser: `http://localhost:10000`

---

## 🔐 Quick Login

**Username:** `crop`  
**Password:** `crop`

*(Hardcoded credentials for quick access)*

---

## 📂 Project Structure

```
CropSpectra/
│
├── venv/                       # Virtual environment (pre-configured)
├── app.py                      # Main Flask application
├── classes.json                # Crop disease class names
├── disease_info.json           # Disease details
├── crop_disease_model.h5       # Trained ML model
├── train_features.pkl          # Feature vectors
├── train_labels.pkl            # Training labels
├── users.db                    # SQLite database (auto-created)
├── requirements.txt            # Python dependencies
├── feedback.csv                # User feedback data
│
├── templates/                  # HTML templates
│   ├── home.html
│   ├── login.html
│   ├── signup.html
│   ├── predict.html
│   ├── about.html
│   ├── blogs.html
│   └── contact.html
│
└── static/
    └── uploads/                # Uploaded images & audio files
```

---

## 🐛 Troubleshooting

### ❌ **Error: 'venv' is not recognized**

**Solution:**
```cmd
# Make sure you're in the correct directory
cd c:\Users\abhin\Downloads\Abhi\Abhi
dir  # Check if 'venv' folder exists
```

---

### ❌ **Error: Script execution disabled (Windows)**

**Solution:**
```cmd
# Run PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then activate again
venv\Scripts\activate
```

---

### ❌ **Error: Permission denied (Linux/Mac)**

**Solution:**
```bash
chmod +x venv/bin/activate
source venv/bin/activate
```

---

### ❌ **Port 10000 already in use**

**Solution:** Edit `app.py` (last line):
```python
port = int(os.environ.get("PORT", 5000))  # Change to 5000 or any free port
```

---

### ❌ **Missing files error**

**Solution:** Make sure these files exist:
```
✅ crop_disease_model.h5
✅ classes.json
✅ disease_info.json
✅ train_features.pkl
✅ train_labels.pkl
```

---

## 🛠️ Technologies Used

| Technology | Version | Purpose |
|-----------|---------|---------|
| Flask | 2.3.3 | Web framework |
| TensorFlow | 2.13.0 | Deep learning |
| deep-translator | 1.11.4 | Translation |
| gTTS | 2.3.2 | Text-to-speech |
| bcrypt | 4.0.1 | Password hashing |

---

## 📝 How to Use

1. **Login** - Use `crop`/`crop` or create a new account
2. **Upload Image** - Go to "Predict" page and upload crop leaf image
3. **View Results** - Get disease prediction with details
4. **Text-to-Speech** - Click speaker button to listen
5. **Translate** - Select language (Hindi/Bengali/Marathi/Bhojpuri) and translate

---

## 🌱 Supported Crops & Diseases (27+ Classes)

- **Wheat** (Brown Rust, Yellow Rust, Septoria, Loose Smut, Healthy)
- **Rice** (Bacterial Leaf Blight, Brown Spot, Leaf Blast, Healthy)
- **Maize / Corn** (Common Rust, Gray Leaf Spot, Northern Leaf Blight, Healthy)
- **Tomato** (Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy)
- **Potato** (Early Blight, Late Blight, Healthy)
- **Bell Pepper** (Bacterial Spot, Healthy)
- **Intelligent Out-of-Distribution (OOD)**: Automatically detects non-plant images / untrained classes and outputs *"Disease Not Found"* with farmer guidance.

---

## 🔄 Stop Application

Press `Ctrl+C` in terminal

To deactivate virtual environment:
```bash
deactivate
```

---

## 📧 Contact & Support

📧 **Email:** cropspectra@gmail.com  
🌍 **Website:** [CropSpectra](#)

---

## 🙏 Acknowledgments

- PlantVillage Dataset
- TensorFlow & Keras Community
- Flask Framework
- Google Translate API

---

## ⚠️ Important Notes

- Virtual environment (`venv`) folder is **pre-configured** with all dependencies
- **Do not delete** `venv` folder
- Works on **Windows only** (for Linux/Mac, recreate venv)
- Ensure Python 3.8+ is installed on your system

---

**© 2026 CropSpectra | Smart Vision for a Healthier Harvest 🌿**
