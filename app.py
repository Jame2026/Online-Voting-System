# app.py
import os
import io
import uuid
import base64
import hashlib
import json
import cv2
import numpy as np
from PIL import Image
from typing import Optional, Dict, Any
from mysql.connector import Error

from fastapi import FastAPI, Request, Form, UploadFile, File, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from passlib.hash import bcrypt
from deepface import DeepFace  # keep existing DeepFace for other features

# NEW library for face verification (only for vote)
import face_recognition

from db import get_db_connection  # your DB connection function

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="some-secret")
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

# ------------------- HELPERS -------------------
def sha256_hex(data: bytes):
    return hashlib.sha256(data).hexdigest()

def image_bytes_to_array(img_bytes: bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(img)

# For DeepFace features (unchanged)
def compute_embedding_from_array(img_array):
    try:
        embedding = DeepFace.represent(img_array, model_name="Facenet", enforce_detection=True)[0]["embedding"]
        return np.array(embedding, dtype='float32')
    except Exception:
        return None

# ------------------- NEW FACE_RECOGNITION HELPERS FOR VOTE -------------------
SIM_THRESHOLD = 0.55  # cosine similarity threshold for face matching

def compute_face_encoding(img_array):
    encs = face_recognition.face_encodings(img_array)
    if not encs:
        return None
    return encs[0]

def cosine_similarity(a: np.ndarray, b: np.ndarray):
    if a is None or b is None: return 0.0
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0: return 0.0
    return float(np.dot(a, b) / denom)

# ------------------- PAGES -------------------
@app.get("/", response_class=HTMLResponse)
@app.get("/home", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("home.html", {"request": request, "message": ""})

@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "message": ""})

# ------------------- LOGIN -------------------
@app.post("/login")
async def login(request: Request,
                username: str = Form(...),
                password: str = Form(...),
                phone: str = Form(None),
                selfie: UploadFile = File(None)):

    if not selfie:
        return templates.TemplateResponse("login.html", {"request": request, "message": "Please capture a selfie."})

    selfie_bytes = await selfie.read()
    img_array = image_bytes_to_array(selfie_bytes)
    probe_emb = compute_embedding_from_array(img_array)
    if probe_emb is None:
        return templates.TemplateResponse("login.html", {"request": request, "message": "No face detected."})

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        cursor.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
        user = cursor.fetchone()

        if user:
            if phone:
                cursor.execute("UPDATE users SET phone=%s WHERE id=%s", (phone, user['id']))
                conn.commit()

            request.session['user_id'] = user["id"]
            return RedirectResponse(url="/vote", status_code=303)

        else:
            selfie_b64 = base64.b64encode(selfie_bytes).decode('utf-8')
            cursor.execute(
                "INSERT INTO users (username, password, phone, selfie_base64) VALUES (%s,%s,%s,%s)",
                (username, password, phone, selfie_b64)
            )
            conn.commit()
            cursor.execute("SELECT * FROM users WHERE username=%s", (username,))
            user = cursor.fetchone()
            if not user:
                raise HTTPException(status_code=500, detail="Failed to create user")

            request.session['user_id'] = user["id"]
            return RedirectResponse(url="/vote", status_code=303)

    finally:
        cursor.close()
        conn.close()

# ------------------- ENROLL -------------------
@app.post("/enroll")
def enroll(payload: dict):
    username = payload.get("username")
    password = payload.get("password")
    email = payload.get("email")
    phone = payload.get("phone")
    comment = payload.get("comment")
    selfie_b64 = payload.get("selfie_base64")

    if not username or not password or not email or not selfie_b64:
        raise HTTPException(status_code=400, detail="Missing fields")

    # Decode selfie to image array
    try:
        selfie_bytes = base64.b64decode(selfie_b64.split(",")[-1])
        nparr = np.frombuffer(selfie_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        new_encodings = face_recognition.face_encodings(rgb_img)
        if not new_encodings:
            raise HTTPException(status_code=400, detail="No face detected in the selfie")
        new_encoding = new_encodings[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid selfie image: {str(e)}")

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        # Check username/email
        cursor.execute("SELECT id FROM users WHERE username=%s OR email=%s", (username, email))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Username or email already exists")

        # Check against existing faces to prevent duplicate enrollment
        cursor.execute("SELECT selfie_base64 FROM users")
        for row in cursor.fetchall():
            existing_b64 = row['selfie_base64']
            if not existing_b64: 
                continue
            existing_bytes = base64.b64decode(existing_b64.split(",")[-1])
            existing_img = cv2.imdecode(np.frombuffer(existing_bytes, np.uint8), cv2.IMREAD_COLOR)
            existing_rgb = cv2.cvtColor(existing_img, cv2.COLOR_BGR2RGB)
            existing_encs = face_recognition.face_encodings(existing_rgb)
            if not existing_encs:
                continue
            existing_encoding = existing_encs[0]

            match = face_recognition.compare_faces([existing_encoding], new_encoding, tolerance=0.55)
            if match[0]:
                raise HTTPException(status_code=400, detail="This face is already enrolled")

        # Insert new user
        cursor.execute(
            "INSERT INTO users (username, password, email, phone, comment, selfie_base64) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (username, password, email, phone, comment, selfie_b64)
        )
        conn.commit()
        return JSONResponse({"status": "ok"})
    finally:
        cursor.close()
        conn.close()

# ------------------- VOTE -------------------
@app.post("/vote")
def vote(request: Request, candidate_id: int = Form(...)):
    user_id = request.session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Not logged in")

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        # Check if user already voted
        cursor.execute("SELECT * FROM votes WHERE user_id=%s", (user_id,))
        if cursor.fetchone():
            return RedirectResponse(url="/results", status_code=303)

        # Save the vote
        cursor.execute(
            "INSERT INTO votes (user_id, candidate_id) VALUES (%s, %s)",
            (user_id, candidate_id)
        )
        conn.commit()
        return RedirectResponse(url="/results", status_code=303)
    finally:
        cursor.close()
        conn.close()






# ------------------- VERIFY OLD USER -------------------
@app.get("/verify_old_user")
def verify_old_user_page(request: Request):
    return templates.TemplateResponse("verify_old_user.html", {"request": request})

@app.post("/vote_request")
def vote_request(request: Request, email: str = Form(...)):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT id FROM users WHERE email=%s", (email,))
        user = cursor.fetchone()
        if not user:
            raise HTTPException(status_code=404, detail="Email not found. Please enroll first.")

        request.session['user_id'] = user['id']
        return RedirectResponse(url="/vote", status_code=303)
    finally:
        cursor.close()
        conn.close()

# ------------------- VOTE -------------------
@app.get("/vote", response_class=HTMLResponse)
def vote_page(request: Request):
    user_id = request.session.get("user_id")
    if not user_id:
        return RedirectResponse("/login")
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT id, name, party, photo FROM candidates")
        candidates = cursor.fetchall()
        return templates.TemplateResponse("vote.html", {"request": request, "candidates": candidates, "message": ""})
    finally:
        cursor.close()
        conn.close()

# ------------------- RESULTS -------------------
@app.get("/results", response_class=HTMLResponse)
def results(request: Request):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT c.id, c.name, c.party, c.photo, COUNT(v.id) as votes
            FROM candidates c
            LEFT JOIN votes v ON c.id = v.candidate_id
            GROUP BY c.id
            ORDER BY votes DESC
        """)
        results_data = cursor.fetchall()
        return templates.TemplateResponse("results.html", {"request": request, "results": results_data})
    finally:
        cursor.close()
        conn.close()

# ------------------- LOGOUT -------------------
@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)

# ---------------- helpers ----------------
def ensure_static_candidates_dir():
    path = os.path.join(os.getcwd(), "static", "candidates")
    os.makedirs(path, exist_ok=True)
    return path

def require_admin(request: Request):
    if not request.session.get("admin_id"):
        raise HTTPException(status_code=303, headers={"Location": "/admin/login"})
    return request.session["admin_id"]

# ---------------- admin login/logout/dashboard ----------------
@app.get("/admin/login", response_class=HTMLResponse)
def admin_login_form(request: Request):
    return templates.TemplateResponse("admin_login.html", {"request": request, "message": ""})

@app.post("/admin/login")
def admin_login(request: Request, username: str = Form(...), password: str = Form(...)):
    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute("SELECT id, password_hash, role FROM admins WHERE username=%s", (username,))
        admin = cur.fetchone()
        if not admin:
            return templates.TemplateResponse("admin_login.html", {"request": request, "message": "Invalid credentials"})
        # verify bcrypt hash safely
        try:
            if not bcrypt.verify(password, admin["password_hash"]):
                return templates.TemplateResponse("admin_login.html", {"request": request, "message": "Invalid credentials"})
        except ValueError:
            return templates.TemplateResponse("admin_login.html", {"request": request, "message": "Server configuration error. Contact admin."})

        request.session['admin_id'] = admin['id']
        request.session['admin_role'] = admin.get('role', 'admin')
        return RedirectResponse("/admin/dashboard", status_code=303)
    finally:
        cur.close()
        conn.close()

@app.get("/admin/logout")
def admin_logout(request: Request):
    request.session.pop('admin_id', None)
    request.session.pop('admin_role', None)
    return RedirectResponse("/admin/login", status_code=303)

# ---------------- admin dashboard ----------------
@app.get("/admin/dashboard", response_class=HTMLResponse)
def admin_dashboard(request: Request, admin_id: int = Depends(require_admin)):
    try:
        conn = get_db_connection()
        cur = conn.cursor(dictionary=True)

        # Totals
        cur.execute("SELECT COUNT(*) AS total_voters FROM users")
        total_voters = cur.fetchone()["total_voters"] or 0

        cur.execute("SELECT COUNT(*) AS total_candidates FROM candidates")
        total_candidates = cur.fetchone()["total_candidates"] or 0

        cur.execute("SELECT COUNT(*) AS total_votes FROM votes")
        total_votes = cur.fetchone()["total_votes"] or 0

        # Candidates with vote counts
        cur.execute("""
            SELECT c.id, c.name, c.photo, COALESCE(COUNT(v.id), 0) AS votes
            FROM candidates c
            LEFT JOIN votes v ON v.candidate_id = c.id
            GROUP BY c.id, c.name, c.photo
            ORDER BY votes DESC
        """)
        candidates = cur.fetchall()

        # Prepare chart data
        chart_data = [{"name": r["name"], "votes": int(r["votes"])} for r in candidates]

        return templates.TemplateResponse("admin_dashboard.html", {
            "request": request,
            "total_voters": total_voters,
            "total_candidates": total_candidates,
            "total_votes": total_votes,
            "candidates": candidates,
            "chart_data": chart_data
        })
        
    except Exception as e:
        return HTMLResponse(f"<h1>Server Error</h1><pre>{e}</pre>")

    finally:
        try: cur.close()
        except: pass
        try: conn.close()
        except: pass

# ---------------- manage candidates ----------------
@app.get("/admin/candidates", response_class=HTMLResponse)
def admin_candidates_list(request: Request, admin_id: int = Depends(require_admin)):
    conn = get_db_connection(); cur = conn.cursor(dictionary=True)
    try:
        cur.execute("SELECT id, name, party, photo, created_at FROM candidates ORDER BY created_at DESC")
        rows = cur.fetchall()
        return templates.TemplateResponse("admin_candidates.html", {"request": request, "candidates": rows})
    finally:
        cur.close(); conn.close()

@app.post("/admin/candidates/create")
async def admin_create_candidate(
    request: Request,
    name: str = Form(...),
    party: str = Form(None),
    photo_file: UploadFile = File(None),
    admin_id: int = Depends(require_admin)
):
    filename = None
    if photo_file:
        ext = os.path.splitext(photo_file.filename)[1]
        filename = f"{uuid.uuid4().hex}{ext}"
        dest = os.path.join(ensure_static_candidates_dir(), filename)
        contents = await photo_file.read()
        with open(dest, "wb") as f:
            f.write(contents)
    conn = get_db_connection(); cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO candidates (name, party, photo, created_at) VALUES (%s,%s,%s,NOW())",
            (name, party, filename)
        )
        conn.commit()
        return RedirectResponse("/admin/candidates", status_code=303)
    finally:
        cur.close(); conn.close()

@app.get("/admin/candidates/{candidate_id}/edit", response_class=HTMLResponse)
def admin_edit_candidate_form(request: Request, candidate_id: int, admin_id: int = Depends(require_admin)):
    conn = get_db_connection(); cur = conn.cursor(dictionary=True)
    try:
        cur.execute("SELECT id, name, party, photo FROM candidates WHERE id=%s", (candidate_id,))
        c = cur.fetchone()
        if not c:
            return RedirectResponse("/admin/candidates", status_code=303)
        return templates.TemplateResponse("admin_edit_candidate.html", {"request": request, "candidate": c})
    finally:
        cur.close(); conn.close()

@app.post("/admin/candidates/{candidate_id}/edit")
async def admin_edit_candidate(
    request: Request,
    candidate_id: int,
    name: str = Form(...),
    party: str = Form(None),
    photo_file: UploadFile = File(None),
    admin_id: int = Depends(require_admin)
):
    conn = get_db_connection(); cur = conn.cursor()
    try:
        if photo_file:
            ext = os.path.splitext(photo_file.filename)[1]
            filename = f"{uuid.uuid4().hex}{ext}"
            dest = os.path.join(ensure_static_candidates_dir(), filename)
            contents = await photo_file.read()
            with open(dest, "wb") as f:
                f.write(contents)
            cur.execute(
                "UPDATE candidates SET name=%s, party=%s, photo=%s WHERE id=%s",
                (name, party, filename, candidate_id)
            )
        else:
            cur.execute(
                "UPDATE candidates SET name=%s, party=%s WHERE id=%s",
                (name, party, candidate_id)
            )
        conn.commit()
        return RedirectResponse("/admin/candidates", status_code=303)
    finally:
        cur.close(); conn.close()

@app.post("/admin/candidates/{candidate_id}/delete")
def admin_delete_candidate(request: Request, candidate_id: int, admin_id: int = Depends(require_admin)):
    conn = get_db_connection(); cur = conn.cursor(dictionary=True)
    try:
        cur.execute("SELECT photo FROM candidates WHERE id=%s", (candidate_id,))
        r = cur.fetchone()
        if r and r.get("photo"):
            path = os.path.join(os.getcwd(), "static", "candidates", r["photo"])
            if os.path.exists(path):
                os.remove(path)
        cur.execute("DELETE FROM candidates WHERE id=%s", (candidate_id,))
        conn.commit()
        return RedirectResponse("/admin/candidates", status_code=303)
    finally:
        cur.close(); conn.close()



# admin_users route
@app.get("/admin/users", response_class=HTMLResponse)
def admin_users(request: Request):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        # Fetch all users
        cursor.execute("SELECT id, username, email, phone, comment, selfie_base64 FROM users")
        users = cursor.fetchall()

        # Add vote status
        for u in users:
            cursor.execute("SELECT * FROM votes WHERE user_id=%s", (u['id'],))
            u['vote_status'] = "✅ Voted" if cursor.fetchone() else "❌ Not Voted"

        return templates.TemplateResponse("admin_users.html", {"request": request, "users": users})
    finally:
        cursor.close()
        conn.close()

@app.post("/admin/users/{user_id}/delete")
def delete_user(user_id: int):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        # Delete user from database
        cursor.execute("DELETE FROM users WHERE id=%s", (user_id,))
        conn.commit()
        # Redirect back to the users page
        return RedirectResponse("/admin/users", status_code=303)
    finally:
        cursor.close()
        conn.close()









def require_admin(request: Request):
    try:
        sess = request.session
    except:
        sess = {}
    if not sess.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin required")

@app.get("/admin/chart_data")
def admin_chart_data(request: Request):
    # require_admin(request)  # uncomment if you need admin auth

    conn = None
    cursor = None
    try:
        conn = get_db_connection()  # <-- use your db.py function
        cursor = conn.cursor(dictionary=True)

        # Query candidates & votes
        cursor.execute("""
            SELECT c.id, c.name, COALESCE(v.votes,0) AS votes
            FROM candidates c
            LEFT JOIN (
                SELECT candidate_id, COUNT(*) AS votes
                FROM votes
                GROUP BY candidate_id
            ) v ON c.id = v.candidate_id
            ORDER BY votes DESC, c.name ASC
        """)
        rows = cursor.fetchall() or []

        # Totals
        cursor.execute("SELECT COUNT(*) AS total_candidates FROM candidates")
        total_candidates = int(cursor.fetchone().get('total_candidates',0) or 0)

        cursor.execute("SELECT COUNT(*) AS total_votes FROM votes")
        total_votes = int(cursor.fetchone().get('total_votes',0) or 0)

        labels = [r['name'] for r in rows]
        data = [int(r['votes'] or 0) for r in rows]
        ids = [int(r['id']) for r in rows]

        return JSONResponse({
            "ok": True,
            "total_candidates": total_candidates,
            "total_votes": total_votes,
            "labels": labels,
            "data": data,
            "ids": ids,
            "rows": rows
        })

    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    finally:
        if cursor:
            try: cursor.close()
            except: pass
        if conn:
            try: conn.close()
            except: pass
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("admin_candidate_chart.html", {"request": request})
