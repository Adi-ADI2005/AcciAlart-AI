# 🚨 AcciAlert-AI — Accident Severity & Casualty Prediction System

AcciAlert-AI is an AI-powered accident analysis system that predicts **accident severity** and **casualty impact** using machine-learning techniques.  
It supports **faster emergency response, early risk assessment, and safety planning**.

---

## ✨ Features

- 🔐 User Registration & Login (MongoDB Auth)
- 🏠 User Dashboard
- 📊 Accident Severity Prediction
- 🧑‍⚕️ Casualty Risk Analysis
- 🧠 ML-Model Powered Backend
- 📈 Result Visualization
- 🗄️ MongoDB User History Support

---

## 🧠 Prediction Capabilities

The system predicts:

- 🚦 Accident Severity Level  
- 🩺 Casualty Risk / Impact  

The ML model is trained on accident dataset features such as:

- Number of vehicles involved  
- Number of casualties  
- Weather condition  
- Road surface condition  

---

## 🛠 Tech Stack

**Frontend**
- HTML, CSS, JavaScript

**Backend**
- Python (Flask)

**Machine Learning**
- Scikit-learn  
- Pandas  
- NumPy  
- SMOTE (for resampling)

**Database**
- MongoDB

---

## 📂 Project Structure

```
AcciAlert-AI/
│
├── static/               # CSS, JS, Images
├── templates/            # HTML Pages
│   ├── register.html
│   ├── login.html
│   ├── home.html
│   ├── prediction.html
│   └── result.html
│
├── RTA.csv               # Dataset
├── model.pkl             # Trained ML Model
├── app.py                # Flask Application
├── requirements.txt      # Dependencies
└── README.md
```

---

## 🚀 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Adi-ADI2005/AcciAlert-AI.git
cd AcciAlert-AI
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Train / Load ML Model (optional)

If you modify dataset or features, retrain the model and save `model.pkl`.

### 4️⃣ Run the Application

```bash
python app.py
```

App runs at:

```
http://127.0.0.1:5000/
```

---

## 🧠 Workflow

1️⃣ User registers & logs in  
2️⃣ Enters accident-related parameters  
3️⃣ ML model processes inputs  
4️⃣ System predicts:

- Accident Severity  
- Casualty Risk Level  

5️⃣ Results displayed on result page

---

## 📌 Use Cases

- 🚑 Emergency Response Planning  
- 🛣 Road Safety Analytics  
- 🧮 Accident Risk Assessment  
- 🏙 Smart City Traffic Systems  

---

## 🔮 Future Enhancements

- Real-time accident detection
- GPS / IoT integration
- Automatic emergency alerting
- Mobile app support
- Admin analytics dashboard

---

## 👨‍💻 Author

**Aditya Mishra**

⭐ If you found this project useful — consider starring the repo!

---

