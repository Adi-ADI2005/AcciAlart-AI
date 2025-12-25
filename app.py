from flask import Flask, render_template, request, redirect, url_for, flash, session
from pymongo import MongoClient
import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

app = Flask(__name__)
app.secret_key = "secret_key_here"

# -------------------- MONGODB CONNECTION --------------------
client = MongoClient("mongodb://localhost:27017/")
db = client.accident
users_collection = db.users


# -------------------------------------------------------------
#                     REGISTRATION
# -------------------------------------------------------------
@app.route('/', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        mobile = request.form.get('mobile')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        captcha_input = request.form.get('captcha_input')
        captcha_answer = request.form.get('captcha_answer')

        if len(mobile) != 10:
            flash("Mobile number must be 10 digits", "error")
            return render_template('reg.html')

        if password != confirm_password:
            flash("Passwords do not match", "error")
            return render_template('reg.html')

        if captcha_input != captcha_answer:
            flash("Incorrect CAPTCHA", "error")
            return render_template('reg.html')

        if users_collection.find_one({"mobile": mobile}):
            flash("Mobile already registered!", "error")
            return render_template('reg.html')

        users_collection.insert_one({
            "first_name": first_name,
            "last_name": last_name,
            "mobile": mobile,
            "password": password
        })

        flash("Registration successful!", "success")
        return redirect(url_for('login'))

    return render_template('reg.html')


# -------------------------------------------------------------
#                     LOGIN
# -------------------------------------------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == "POST":
        mobile = request.form.get("mobile")
        password = request.form.get("password")
        captcha_input = request.form.get("captcha_input")
        captcha_answer = request.form.get("captcha_answer")

        if captcha_input != captcha_answer:
            flash("Invalid CAPTCHA", "danger")
            return redirect(url_for("login"))

        user = users_collection.find_one({"mobile": mobile})

        if user and user["password"] == password:
            session["user"] = user["first_name"]
            flash("Login successful!", "success")
            return redirect(url_for("index"))
        else:
            flash("Invalid mobile or password", "danger")
            return redirect(url_for("login"))

    return render_template("login.html")


# -------------------------------------------------------------
#                 FORGOT PASSWORD
# -------------------------------------------------------------
@app.route('/forgot_password', methods=['POST'])
def forgot_password():
    mobile = request.form.get('mobile', '')
    password = request.form.get('password', '')
    confirm_password = request.form.get('confirm_password', '')

    if not re.fullmatch(r'\d{10}', mobile):
        flash("Invalid 10-digit mobile number!", "error")
        return redirect(url_for('index'))

    if password != confirm_password:
        flash("Passwords do not match!", "error")
        return redirect(url_for('index'))

    user = users_collection.find_one({'mobile': mobile})
    if user:
        users_collection.update_one(
            {'mobile': mobile},
            {'$set': {'password': password}}
        )
        flash("Password reset successful!", "success")
    else:
        flash("Mobile not registered!", "error")

    return redirect(url_for('login'))


# -------------------------------------------------------------
#                     HOME PAGE
# -------------------------------------------------------------
@app.route('/index')
def index():
    if "user" not in session:
        flash("Please login first.", "warning")
        return redirect(url_for("login"))
    return render_template("index.html", username=session["user"])


@app.route('/logout')
def logout():
    session.pop("user", None)
    flash("Logged out successfully.", "info")
    return redirect(url_for("login"))





# -------------------------------------------------------------
#            SEVERITY PREDICTION HOME PAGE
# -------------------------------------------------------------
@app.route('/severity')
def severity_page():
    if "user" not in session:
        flash("Please log in first.", "warning")
        return redirect(url_for("login"))
    return render_template('Severity.html')


# -------------------------------------------------------------
#       PREDICT ACCIDENT + CASUALTY SEVERITY (ML MODEL)
# -------------------------------------------------------------
@app.route('/predict_severity', methods=['POST'])
def predict_severity():

    # ---- Load Dataset ----
    df = pd.read_csv("rta.csv")

    features = [
        "Area_accident_occured",
        "Lanes_or_Medians",
        "Road_allignment",
        "Types_of_Junction",
        "Road_surface_type",
        "Road_surface_conditions",
        "Light_conditions",
        "Type_of_collision",
        "Number_of_vehicles_involved"
    ]

    X = df[features].copy()

    y1 = df["Casualty_severity"]     # (1=Normal, 2=Minor, 3=Major)
    y2 = df["Accident_severity"]     # (0=Normal, 1=Minor, 2=Major)

    # ---- Encode All Columns (with Unknown label support) ----
    encoders = {}
    for col in X.columns:
        encoders[col] = LabelEncoder()

        X[col] = X[col].fillna("Unknown").astype(str)

        # Fit encoder including Unknown
        encoders[col].fit(list(X[col].unique()) + ["Unknown"])

        X[col] = encoders[col].transform(X[col])

    sm = SMOTE()

    # ---- Train Model 1 (Casualty) ----
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y1, test_size=0.2, random_state=42)
    X_train1_res, y_train1_res = sm.fit_resample(X_train1, y_train1)

    model1 = RandomForestClassifier()
    model1.fit(X_train1_res, y_train1_res)

    # ---- Train Model 2 (Accident) ----
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y2, test_size=0.2, random_state=42)
    X_train2_res, y_train2_res = sm.fit_resample(X_train2, y_train2)

    model2 = RandomForestClassifier()
    model2.fit(X_train2_res, y_train2_res)

    # ---- Get User Input ----
    input_data = {
        "Area_accident_occured": request.form["area_accident_occurred"],
        "Lanes_or_Medians": request.form["lanes_or_medians"],
        "Road_allignment": request.form["road_alignment"],
        "Types_of_Junction": request.form["types_of_junction"],
        "Road_surface_type": request.form["road_surface_type"],
        "Road_surface_conditions": request.form["road_surface_conditions"],
        "Light_conditions": request.form["light_conditions"],
        "Type_of_collision": request.form["type_of_collision"],
        "Number_of_vehicles_involved": request.form["number_of_vehicles_involved"]
    }

    new_df = pd.DataFrame([input_data])

    # ---- Apply encoding to new input ----
    for col in new_df.columns:
        new_df[col] = new_df[col].astype(str)
        val = new_df[col].iloc[0]

        if val not in encoders[col].classes_:
            new_df[col] = "Unknown"

        new_df[col] = encoders[col].transform(new_df[col])

    # ---- Predict (numeric output) ----
    pred_casualty_num = model1.predict(new_df)[0]
    pred_accident_num = model2.predict(new_df)[0]

    # ---- Mapping numeric → text ----
    accident_map = {0:"Minor" , 1:"Major" , 2:"Fatal" }
    casualty_map = {1: "Minor", 2:"Major" , 3:"Fatal" }

    pred_accident = accident_map.get(pred_accident_num, "Unknown")
    pred_casualty = casualty_map.get(pred_casualty_num, "Unknown")

    # ---- Return Output ----
    return render_template(
        "result.html",
        casualty=pred_casualty,
        accident=pred_accident
    )




# -------------------------------------------------------------
#                     RUN APP
# -------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
