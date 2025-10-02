from flask import Flask, render_template, request, redirect, url_for, session, flash
import requests
import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Change in production!

# --- MySQL Connection Helper ---
def get_db_connection():
    conn = mysql.connector.connect(
        host="localhost",
        user="newsuser",
        password="NewsPass123!",
        database="newsdb"
    )
    return conn

# Use the correct FastAPI URL
API_URL = "http://127.0.0.1:8000/predict/"  # Make sure FastAPI is running here

# ---------------- Routes ---------------- #

# ---------- Home / Prediction ----------
@app.route("/", methods=["GET", "POST"])
def home():
    if "user_id" not in session:
        return redirect(url_for("login"))

    result = None
    if request.method == "POST":
        headline = request.form.get("headline")
        if headline:
            try:
                response = requests.post(API_URL, json={"headline": headline})
                response.raise_for_status()
                data = response.json()
                confidence = float(data.get("confidence", 0))
                result = {
                    "category": data.get("predicted_category", "Unknown"),
                    "confidence": round(confidence, 4)
                }

                # Save to DB (user-specific)
                conn = get_db_connection()
                cursor = conn.cursor(dictionary=True)
                cursor.execute(
                    "INSERT INTO predictions (headline, predicted_category, confidence, user_id) VALUES (%s, %s, %s, %s)",
                    (headline, result["category"], result["confidence"], session["user_id"])
                )
                conn.commit()
                cursor.close()
                conn.close()

            except requests.exceptions.RequestException as e:
                result = {"error": f"API request failed: {e}"}
            except Exception as e:
                result = {"error": str(e)}

    return render_template("index.html", result=result)

# ---------- Registration ----------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        hashed_pw = generate_password_hash(password)
        try:
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "INSERT INTO users (email, password) VALUES (%s, %s)",
                (email, hashed_pw)
            )
            conn.commit()
            cursor.close()
            conn.close()
            flash("✅ Registration successful! Please login.", "success")
            return redirect(url_for("login"))
        except mysql.connector.Error as e:
            flash(f"❌ Error: {str(e)}", "danger")

    return render_template("register.html")

# ---------- Login ----------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if user and check_password_hash(user["password"], password):
            session["user_id"] = user["id"]
            session["email"] = user["email"]
            flash("✅ Login successful!", "success")
            return redirect(url_for("home"))
        else:
            flash("❌ Invalid credentials", "danger")

    return render_template("login.html")

# ---------- Logout ----------
@app.route("/logout")
def logout():
    session.clear()
    flash("👋 You have logged out.", "info")
    return redirect(url_for("login"))

# ---------- Prediction History ----------
@app.route("/history")
def history():
    if "user_id" not in session:
        return redirect(url_for("login"))

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT * FROM predictions WHERE user_id = %s ORDER BY created_at DESC",
        (session["user_id"],)
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template("history.html", history=rows)

# ---------- Batch Prediction ----------
@app.route("/batch", methods=["GET", "POST"])
def batch():
    if "user_id" not in session:
        return redirect(url_for("login"))

    results = []
    if request.method == "POST":
        file = request.files.get("file")
        if file:
            import pandas as pd
            df = pd.read_csv(file)
            for headline in df["headline"].tolist():
                try:
                    response = requests.post(API_URL, json={"headline": headline})
                    response.raise_for_status()
                    data = response.json()
                    confidence = float(data.get("confidence", 0))
                    results.append({
                        "headline": headline,
                        "category": data.get("predicted_category", "Unknown"),
                        "confidence": round(confidence, 4)
                    })

                    # Save to DB
                    conn = get_db_connection()
                    cursor = conn.cursor(dictionary=True)
                    cursor.execute(
                        "INSERT INTO predictions (headline, predicted_category, confidence, user_id) VALUES (%s, %s, %s, %s)",
                        (headline, data.get("predicted_category", "Unknown"), round(confidence, 4), session["user_id"])
                    )
                    conn.commit()
                    cursor.close()
                    conn.close()

                except requests.exceptions.RequestException:
                    results.append({"headline": headline, "error": "API request failed"})
                except Exception:
                    results.append({"headline": headline, "error": "Unknown error"})

    return render_template("batch.html", results=results)

# ---------------- Run ---------------- #
if __name__ == "__main__":
    app.run(debug=True)
