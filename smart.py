from collections import deque
from contextlib import closing
import heapq
import math
import os
from pathlib import Path
import secrets
import sqlite3

import pandas as pd
from flask import Flask, flash, redirect, render_template_string, request, session, url_for
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
APP_ENV = os.environ.get("APP_ENV", "development").strip().lower()
secret_key = os.environ.get("FLASK_SECRET_KEY", "").strip()
if APP_ENV == "production" and not secret_key:
    raise RuntimeError("FLASK_SECRET_KEY must be set in production.")
if not secret_key:
    secret_key = secrets.token_hex(32)

app.config.update(
    SECRET_KEY=secret_key,
    MAX_CONTENT_LENGTH=16 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get(
        "FLASK_COOKIE_SECURE", "1" if APP_ENV == "production" else "0"
    ) == "1",
)
DB = Path(os.environ.get("TRAFFIC_DB", Path(__file__).with_name("traffic.db"))).expanduser()
WEATHER_CODES = {"Clear": 0, "Rain": 1, "Fog": 2}

# ---------------- MACHINE LEARNING ----------------

def train_model():
    data = []

    # Synthetic demo training data
    for density in range(10, 101, 5):
        for speed in [10, 20, 30, 40, 50, 60]:
            for weather in [0, 1, 2]:
                for visibility in [2, 5, 8, 10]:
                    score = (
                        (density >= 70)
                        + (speed <= 20)
                        + (weather == 1)
                        + (weather == 2)
                        + (visibility <= 3)
                    )

                    if score >= 4:
                        risk = "High"
                    elif score >= 2:
                        risk = "Medium"
                    else:
                        risk = "Low"

                    data.append([
                        density, speed, weather, visibility, risk
                    ])

    df = pd.DataFrame(
        data,
        columns=["density", "speed", "weather",
                 "visibility", "risk"]
    )

    model = RandomForestClassifier(
        n_estimators=50, random_state=42
    )

    model.fit(df.iloc[:, :4], df["risk"])
    return model


model = train_model()

def predict_risk(density, speed, weather, visibility):
    features = pd.DataFrame([{
        "density": density,
        "speed": speed,
        "weather": WEATHER_CODES.get(weather, 0),
        "visibility": visibility,
    }])
    return model.predict(features)[0]


# ---------------- DATABASE ----------------

def connect_db():
    DB.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with closing(connect_db()) as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("""
            CREATE TABLE IF NOT EXISTS roads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE COLLATE NOCASE
                    CHECK (length(name) <= 80),
                density INTEGER NOT NULL CHECK (density BETWEEN 0 AND 100),
                speed REAL NOT NULL CHECK (speed BETWEEN 0 AND 150),
                weather TEXT NOT NULL CHECK (weather IN ('Clear', 'Rain', 'Fog')),
                visibility INTEGER NOT NULL CHECK (visibility BETWEEN 1 AND 10)
            )
        """)

        count = connection.execute(
            "SELECT COUNT(*) FROM roads"
        ).fetchone()[0]

        if count == 0:
            connection.executemany("""
                INSERT INTO roads
                (name, density, speed, weather, visibility)
                VALUES (?, ?, ?, ?, ?)
            """, [
                ("Road A", 82, 18, "Rain", 2),
                ("Road B", 24, 42, "Clear", 8),
                ("Road C", 55, 31, "Fog", 3),
                ("Road D", 67, 22, "Clear", 7),
            ])

        connection.commit()


# ---------------- DATA STRUCTURES ----------------

# Queue: vehicles waiting at a traffic signal
vehicle_queue = deque(["Car", "Bus", "Bike"])

# Graph: road network
graph = {
    "North": [("East", 3), ("West", 2)],
    "East": [("North", 3), ("South", 2.5)],
    "South": [("East", 2.5), ("West", 2)],
    "West": [("North", 2), ("South", 2)]
}


def congestion(density, speed):
    if density >= 65 or speed < 20:
        return "High"
    elif density >= 35 or speed < 35:
        return "Medium"
    return "Low"


def dijkstra(start, end):
    if start not in graph or end not in graph:
        return None

    distances = {node: float("inf") for node in graph}
    previous = {}
    distances[start] = 0
    pq = [(0, start)]

    while pq:
        cost, node = heapq.heappop(pq)

        if cost != distances[node]:
            continue

        if node == end:
            break

        for neighbor, weight in graph[node]:
            new_cost = cost + weight

            if new_cost < distances[neighbor]:
                distances[neighbor] = new_cost
                previous[neighbor] = node
                heapq.heappush(pq, (new_cost, neighbor))

    if distances[end] == float("inf"):
        return "No route found"

    path = [end]
    while path[-1] != start:
        path.append(previous[path[-1]])

    path.reverse()
    return " → ".join(path) + f" (Cost: {distances[end]})"


# ---------------- WEB DASHBOARD ----------------

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="theme-color" content="#173b37">
    <link rel="icon" type="image/svg+xml" href="{{ url_for('favicon') }}">
    <title>Traffic Monitor</title>
    <style>
        :root {
            color-scheme: light;
            --ink: #183431;
            --muted: #61716d;
            --line: #dce5e1;
            --paper: #f4f7f5;
            --white: #ffffff;
            --green: #167563;
            --green-dark: #173b37;
            --coral: #c8573e;
            --amber: #d4a33b;
            --red: #b83d36;
            --shadow: 0 12px 32px rgba(24, 52, 49, .07);
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            color: var(--ink);
            background: var(--paper);
            font-family: "Segoe UI", Tahoma, sans-serif;
            line-height: 1.5;
        }
        a { color: inherit; }
        .masthead {
            color: var(--white);
            background-color: var(--green-dark);
            background-image: linear-gradient(120deg, rgba(255,255,255,.035) 25%, transparent 25%, transparent 50%, rgba(255,255,255,.035) 50%, rgba(255,255,255,.035) 75%, transparent 75%);
            background-size: 28px 28px;
            border-bottom: 4px solid var(--amber);
        }
        .masthead-inner, main {
            width: min(1180px, calc(100% - 40px));
            margin: 0 auto;
        }
        .masthead-inner {
            min-height: 94px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 24px;
        }
        .brand { display: flex; align-items: center; gap: 14px; }
        .brand-mark {
            width: 42px;
            height: 42px;
            display: grid;
            place-items: center;
            color: var(--green-dark);
            background: var(--amber);
            border-radius: 4px;
            font: 700 18px Georgia, serif;
        }
        .brand h1 {
            margin: 0;
            font: 600 25px Georgia, "Times New Roman", serif;
        }
        .brand p { margin: 3px 0 0; color: #c4d5d0; font-size: 13px; }
        .feed-status {
            padding: 8px 12px;
            border: 1px solid rgba(255,255,255,.32);
            border-radius: 3px;
            color: #ffe4a3;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: .08em;
            white-space: nowrap;
        }
        main { padding: 34px 0 56px; }
        .page-intro {
            display: flex;
            align-items: end;
            justify-content: space-between;
            gap: 20px;
            margin-bottom: 24px;
        }
        .eyebrow {
            margin: 0 0 6px;
            color: var(--green);
            font-size: 11px;
            font-weight: 700;
            letter-spacing: .12em;
            text-transform: uppercase;
        }
        h2, h3 { margin-top: 0; }
        .page-intro h2 {
            margin: 0;
            font: 600 32px Georgia, "Times New Roman", serif;
        }
        .page-intro p:last-child { margin: 8px 0 0; color: var(--muted); }
        .metrics {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 14px;
            margin-bottom: 28px;
        }
        .metric {
            min-height: 108px;
            padding: 18px 20px;
            background: var(--white);
            border: 1px solid var(--line);
            border-top: 3px solid var(--green);
            box-shadow: var(--shadow);
        }
        .metric:nth-child(2) { border-top-color: var(--coral); }
        .metric:nth-child(3) { border-top-color: var(--amber); }
        .metric-label { color: var(--muted); font-size: 13px; }
        .metric-value { display: block; margin-top: 4px; font: 600 30px Georgia, serif; }
        .workspace {
            display: grid;
            grid-template-columns: minmax(0, 1.7fr) minmax(280px, .8fr);
            gap: 30px;
            align-items: start;
        }
        .workspace > * { min-width: 0; }
        .section-heading {
            display: flex;
            align-items: end;
            justify-content: space-between;
            gap: 16px;
            margin-bottom: 14px;
        }
        .section-heading h2, .tool-section h2 {
            margin: 0;
            font: 600 22px Georgia, "Times New Roman", serif;
        }
        .section-heading p { margin: 3px 0 0; color: var(--muted); font-size: 13px; }
        .table-wrap {
            overflow-x: auto;
            background: var(--white);
            border: 1px solid var(--line);
            box-shadow: var(--shadow);
        }
        table { width: 100%; border-collapse: collapse; text-align: left; }
        th, td { padding: 13px 14px; border-bottom: 1px solid var(--line); }
        th {
            color: var(--muted);
            background: #edf2ef;
            font-size: 10px;
            letter-spacing: .08em;
            text-transform: uppercase;
            white-space: nowrap;
        }
        td { font-size: 13px; }
        tbody tr:last-child td { border-bottom: 0; }
        tbody tr:hover { background: #f7faf8; }
        .road-name { font-weight: 700; white-space: nowrap; }
        .tag {
            display: inline-block;
            min-width: 68px;
            padding: 3px 8px;
            border-radius: 2px;
            text-align: center;
            font-size: 11px;
            font-weight: 700;
        }
        .High { color: #8d2e29; background: #fbe8e5; }
        .Medium { color: #875c10; background: #fbf1d9; }
        .Low { color: #17634d; background: #e2f2eb; }
        .tools { display: grid; gap: 24px; }
        .tool-section { padding-bottom: 22px; border-bottom: 1px solid var(--line); }
        .tool-section > p { margin: 7px 0 16px; color: var(--muted); font-size: 13px; }
        .field-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
        .field { display: grid; gap: 5px; min-width: 0; }
        .field.full { grid-column: 1 / -1; }
        label { color: #3d514c; font-size: 12px; font-weight: 650; }
        input, select {
            width: 100%;
            min-height: 40px;
            padding: 8px 10px;
            color: var(--ink);
            background: var(--white);
            border: 1px solid #bdcbc5;
            border-radius: 3px;
            font: inherit;
        }
        input:focus, select:focus, button:focus-visible {
            outline: 3px solid rgba(22, 117, 99, .22);
            outline-offset: 2px;
            border-color: var(--green);
        }
        button {
            min-height: 40px;
            padding: 9px 14px;
            color: var(--white);
            background: var(--green);
            border: 0;
            border-radius: 3px;
            cursor: pointer;
            font: 650 13px "Segoe UI", Tahoma, sans-serif;
        }
        button:hover { background: #105f50; }
        .secondary-button { color: var(--green-dark); background: #e2ece7; }
        .secondary-button:hover { background: #d5e4dc; }
        .form-actions { margin-top: 14px; }
        .inline-fields { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        .queue-line { padding: 12px; background: #edf2ef; border-left: 3px solid var(--amber); }
        .notice {
            margin: 0 0 20px;
            padding: 12px 14px;
            color: #704d12;
            background: #fff5da;
            border-left: 3px solid var(--amber);
            font-size: 13px;
        }
        .messages { display: grid; gap: 8px; margin-bottom: 18px; }
        .message { padding: 10px 12px; border-left: 3px solid var(--green); background: #e5f2ec; }
        .message.error { border-left-color: var(--red); background: #fbe8e5; }
        .empty { padding: 24px; color: var(--muted); text-align: center; }
        footer { margin-top: 32px; color: var(--muted); font-size: 12px; }
        @media (max-width: 820px) {
            .workspace { grid-template-columns: 1fr; }
            .tools { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 24px; }
            .tool-section:last-child { grid-column: 1 / -1; }
        }
        @media (max-width: 560px) {
            .masthead-inner, main { width: min(100% - 28px, 1180px); }
            .masthead-inner { min-height: 82px; }
            .brand h1 { font-size: 20px; }
            .brand p { font-size: 11px; }
            .brand-mark { width: 36px; height: 36px; }
            .feed-status { max-width: 108px; white-space: normal; text-align: center; }
            main { padding-top: 26px; }
            .page-intro { display: block; }
            .page-intro h2 { font-size: 28px; }
            .metrics { gap: 8px; }
            .metric { min-height: 91px; padding: 12px 10px; }
            .metric-label { font-size: 11px; }
            .metric-value { font-size: 25px; }
            .tools { grid-template-columns: 1fr; }
            .tool-section:last-child { grid-column: auto; }
        }
    </style>
</head>
<body>
<header class="masthead">
    <div class="masthead-inner">
        <div class="brand">
            <span class="brand-mark" aria-hidden="true">TM</span>
            <div>
                <h1>Traffic Monitor</h1>
                <p>Road conditions and route planning</p>
            </div>
        </div>
        <div class="feed-status" role="status">DEMO DATA / NO LIVE FEED</div>
    </div>
</header>
<main>
    <section class="page-intro" aria-labelledby="dashboard-title">
        <div>
            <p class="eyebrow">Operations overview</p>
            <h2 id="dashboard-title">Traffic dashboard</h2>
            <p>Sample values are synthetic; new readings are entered manually.</p>
        </div>
    </section>

    <div class="notice">
        No sensor, external API, or real-time data feed is connected. Risk labels are generated from synthetic demo training data and are not real-world forecasts.
    </div>

    {% with messages = get_flashed_messages(with_categories=true) %}
    {% if messages %}
    <div class="messages" role="status" aria-live="polite">
        {% for category, message in messages %}
        <div class="message {{ category }}">{{ message }}</div>
        {% endfor %}
    </div>
    {% endif %}
    {% endwith %}

    <section class="metrics" aria-label="Traffic summary">
        <div class="metric"><span class="metric-label">Roads recorded</span><strong class="metric-value">{{ roads|length }}</strong></div>
        <div class="metric"><span class="metric-label">High congestion</span><strong class="metric-value">{{ high_count }}</strong></div>
        <div class="metric"><span class="metric-label">Vehicles waiting</span><strong class="metric-value">{{ queue|length }}</strong></div>
    </section>

    <div class="workspace">
        <section aria-labelledby="roads-title">
            <div class="section-heading">
                <div>
                    <h2 id="roads-title">Road conditions</h2>
                    <p>Stored readings and model estimates</p>
                </div>
            </div>
            <div class="table-wrap">
                <table>
                    <thead>
                        <tr><th>Road</th><th>Density</th><th>Speed</th><th>Weather</th><th>Congestion</th><th>Risk</th><th>Green time</th></tr>
                    </thead>
                    <tbody>
                    {% for road in roads %}
                        <tr>
                            <td class="road-name">{{ road.name }}</td>
                            <td>{{ road.density }}%</td>
                            <td>{{ road.speed }} km/h</td>
                            <td>{{ road.weather }}</td>
                            <td><span class="tag {{ road.congestion }}">{{ road.congestion }}</span></td>
                            <td><span class="tag {{ road.risk }}">{{ road.risk }}</span></td>
                            <td>{{ road.green }} sec</td>
                        </tr>
                    {% else %}
                        <tr><td class="empty" colspan="7">No road readings have been recorded.</td></tr>
                    {% endfor %}
                    </tbody>
                </table>
            </div>
        </section>

        <aside class="tools" aria-label="Traffic tools">
            <section class="tool-section" aria-labelledby="add-title">
                <p class="eyebrow">Manual entry</p>
                <h2 id="add-title">Add a reading</h2>
                <p>Enter a road observation to store it in the local database.</p>
                <form method="post" action="{{ url_for('add_road') }}">
                    <input type="hidden" name="csrf_token" value="{{ csrf_token }}">
                    <div class="field-grid">
                        <div class="field full"><label for="name">Road name</label><input id="name" name="name" maxlength="80" autocomplete="off" required></div>
                        <div class="field"><label for="density">Density (%)</label><input id="density" name="density" type="number" min="0" max="100" step="1" required></div>
                        <div class="field"><label for="speed">Speed (km/h)</label><input id="speed" name="speed" type="number" min="0" max="150" step="0.5" required></div>
                        <div class="field"><label for="weather">Weather</label><select id="weather" name="weather" required><option>Clear</option><option>Rain</option><option>Fog</option></select></div>
                        <div class="field"><label for="visibility">Visibility (1-10)</label><input id="visibility" name="visibility" type="number" min="1" max="10" step="1" required></div>
                    </div>
                    <div class="form-actions"><button type="submit">Save reading</button></div>
                </form>
            </section>

            <section class="tool-section" aria-labelledby="route-title">
                <p class="eyebrow">Road network</p>
                <h2 id="route-title">Find a route</h2>
                <p>Choose two known nodes to calculate the lowest-cost path.</p>
                <form method="post" action="{{ url_for('route') }}">
                    <input type="hidden" name="csrf_token" value="{{ csrf_token }}">
                    <div class="inline-fields">
                        <div class="field"><label for="start">Start</label><select id="start" name="start">{% for node in nodes %}<option value="{{ node }}">{{ node }}</option>{% endfor %}</select></div>
                        <div class="field"><label for="end">Destination</label><select id="end" name="end">{% for node in nodes %}<option value="{{ node }}">{{ node }}</option>{% endfor %}</select></div>
                    </div>
                    <div class="form-actions"><button class="secondary-button" type="submit">Calculate route</button></div>
                </form>
                {% if route %}<p><strong>Suggested route:</strong> {{ route }}</p>{% endif %}
            </section>

            <section class="tool-section" aria-labelledby="queue-title">
                <p class="eyebrow">Signal queue</p>
                <h2 id="queue-title">Waiting vehicles</h2>
                <p class="queue-line">{{ queue|join(", ") or "No vehicles waiting" }}</p>
                <form method="post" action="{{ url_for('serve_vehicle') }}">
                    <input type="hidden" name="csrf_token" value="{{ csrf_token }}">
                    <button class="secondary-button" type="submit">Process next vehicle</button>
                </form>
                {% if served %}<p>Processed: <strong>{{ served }}</strong></p>{% endif %}
            </section>
        </aside>
    </div>
    <footer>Traffic Monitor · Local demonstration · Data is stored in SQLite on this machine.</footer>
</main>
</body>
</html>
"""
@app.route("/")
def home():
    with closing(connect_db()) as connection:
        rows = connection.execute(
            "SELECT * FROM roads ORDER BY name"
        ).fetchall()

    roads = []
    for row in rows:
        risk = predict_risk(row["density"], row["speed"], row["weather"], row["visibility"])
        level = congestion(row["density"], row["speed"])

        roads.append({
            "name": row["name"],
            "density": row["density"],
            "speed": row["speed"],
            "weather": row["weather"],
            "risk": risk,
            "congestion": level,
            "green": max(20, min(90, int(20 + row["density"] * 0.7)))
        })

    high_count = sum(r["congestion"] == "High" for r in roads)

    csrf_token = session.setdefault("csrf_token", secrets.token_urlsafe(32))
    return render_template_string(
        HTML,
        roads=roads,
        high_count=high_count,
        nodes=list(graph.keys()),
        route=request.args.get("route", ""),
        queue=list(vehicle_queue),
        served=request.args.get("served", ""),
        csrf_token=csrf_token,
    )


@app.route("/add", methods=["POST"])
def add_road():
    try:
        name = request.form.get("name", "").strip()
        density = int(request.form.get("density", ""))
        speed = float(request.form.get("speed", ""))
        weather = request.form.get("weather", "")
        visibility = int(request.form.get("visibility", ""))

        if not name or len(name) > 80 or any(ord(character) < 32 for character in name):
            raise ValueError("Road name must contain 1 to 80 printable characters.")
        if not 0 <= density <= 100:
            raise ValueError("Density must be from 0 to 100.")
        if not math.isfinite(speed) or not 0 <= speed <= 150:
            raise ValueError("Speed must be from 0 to 150.")
        if weather not in WEATHER_CODES:
            raise ValueError("Choose a listed weather condition.")
        if not 1 <= visibility <= 10:
            raise ValueError("Visibility must be from 1 to 10.")

        with closing(connect_db()) as connection:
            connection.execute("""
                INSERT INTO roads (name, density, speed, weather, visibility)
                VALUES (?, ?, ?, ?, ?)
            """, (name, density, speed, weather, visibility))
            connection.commit()

    except sqlite3.IntegrityError:
        flash("A road with that name already exists.", "error")
        return redirect(url_for("home"))
    except (ValueError, KeyError) as error:
        flash(str(error), "error")
        return redirect(url_for("home"))

    flash("Road reading saved.", "success")
    return redirect(url_for("home"))


@app.route("/route", methods=["POST"])
def route():
    start = request.form.get("start", "")
    end = request.form.get("end", "")
    result = dijkstra(start, end)
    if result is None:
        flash("Choose valid start and destination nodes.", "error")
        return redirect(url_for("home"))
    return redirect(url_for("home", route=result))


@app.route("/serve", methods=["POST"])
def serve_vehicle():
    served = vehicle_queue.popleft() if vehicle_queue else "No vehicles waiting"
    return redirect(url_for("home", served=served))


@app.route("/favicon.ico")
def favicon():
    icon = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
        <rect width="64" height="64" rx="8" fill="#173b37"/>
        <path d="M8 49h48" stroke="#d4a33b" stroke-width="4"/>
        <text x="32" y="39" fill="#ffffff" font-family="Georgia,serif" font-size="25" font-weight="700" text-anchor="middle">TM</text>
    </svg>"""
    return app.response_class(icon, mimetype="image/svg+xml")


@app.get("/healthz")
def health_check():
    try:
        with closing(connect_db()) as connection:
            connection.execute("SELECT 1")
    except sqlite3.Error:
        return {"status": "unhealthy"}, 503
    return {"status": "ok"}, 200


@app.before_request
def protect_state_changes():
    if request.method == "POST":
        expected = session.get("csrf_token", "")
        supplied = request.form.get("csrf_token", "")
        if not expected or not secrets.compare_digest(expected, supplied):
            return "Invalid or expired form token. Reload the page and try again.", 400


@app.after_request
def add_security_headers(response):
    response.headers.setdefault("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; form-action 'self'; base-uri 'self'; object-src 'none'; frame-ancestors 'none'")
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("X-Frame-Options", "DENY")
    return response


initialize_database()

if __name__ == "__main__":
    app.run(host="127.0.0.1", debug=False)
