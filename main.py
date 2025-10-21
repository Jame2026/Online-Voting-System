from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from database import get_db_connection
from starlette.middleware.sessions import SessionMiddleware

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="some-secret")  # for session handling

templates = Jinja2Templates(directory="templates")

# ------------------- LOGIN -------------------
@app.get("/", response_class=HTMLResponse)
@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "message": ""})

@app.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    cursor.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
    user = cursor.fetchone()

    if user:
        if user["has_voted"]:
            cursor.close()
            conn.close()
            return templates.TemplateResponse("login.html", {"request": request, "message": "You already voted!"})
        else:
            request.session['user_id'] = user["user_id"]
            cursor.close()
            conn.close()
            return RedirectResponse(url="/vote", status_code=303)
    else:
        # Create new user
        cursor.execute("INSERT INTO users (username, password, has_voted) VALUES (%s, %s, 0)", (username, password))
        conn.commit()
        cursor.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
        user = cursor.fetchone()
        request.session['user_id'] = user["user_id"]
        cursor.close()
        conn.close()
        return RedirectResponse(url="/vote", status_code=303)

# ------------------- VOTE -------------------
@app.get("/vote", response_class=HTMLResponse)
def vote_page(request: Request):
    user_id = request.session.get("user_id")
    if not user_id:
        return RedirectResponse("/login")

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT candidate_id, name FROM candidates")
    candidates = cursor.fetchall()
    cursor.close()
    conn.close()
    return templates.TemplateResponse("vote.html", {"request": request, "candidates": candidates, "message": ""})

@app.post("/vote")
def submit_vote(request: Request, candidate_id: int = Form(...)):
    user_id = request.session.get("user_id")
    if not user_id:
        return RedirectResponse("/login")

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # Check if already voted
    cursor.execute("SELECT has_voted FROM users WHERE user_id=%s", (user_id,))
    user = cursor.fetchone()
    if user["has_voted"]:
        cursor.close()
        conn.close()
        return RedirectResponse("/results", status_code=303)

    # Insert vote
    cursor.execute("INSERT INTO votes (user_id, candidate_id) VALUES (%s, %s)", (user_id, candidate_id))
    cursor.execute("UPDATE users SET has_voted=1 WHERE user_id=%s", (user_id,))
    cursor.execute("UPDATE candidates SET votes = votes + 1 WHERE candidate_id=%s", (candidate_id,))
    conn.commit()
    cursor.close()
    conn.close()
    return RedirectResponse("/results", status_code=303)

# ------------------- RESULTS -------------------
@app.get("/results", response_class=HTMLResponse)
def results(request: Request):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT candidate_id, name, votes FROM candidates ORDER BY votes DESC")
    results_data = cursor.fetchall()
    cursor.close()
    conn.close()
    return templates.TemplateResponse("results.html", {"request": request, "results": results_data})

# ------------------- LOGOUT -------------------
@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)
