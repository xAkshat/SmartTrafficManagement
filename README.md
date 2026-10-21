# Smart Traffic Management

A local web dashboard for exploring road conditions, estimating congestion and
traffic risk, and demonstrating basic route-planning and signal-queue
algorithms. The project is a self-contained demo: it uses sample readings and
does not connect to live traffic or weather services.

## Features

- View road readings with congestion and traffic-risk estimates.
- Add road observations with density, speed, weather, and visibility values.
- Store readings locally in SQLite. The app creates `traffic.db` and seeds
  sample roads on first launch.
- Find a lowest-cost route between the North, East, South, and West nodes using
  Dijkstra's algorithm.
- Process a first-in, first-out demo queue of waiting vehicles.
- Check service availability at `/healthz`.

Traffic-risk estimates are produced by a Random Forest classifier trained at
startup on synthetic demo data. Congestion labels, route costs, and the
vehicle queue are illustrative and are not intended for real-world traffic
control.

## Requirements

- Python 3.10 or newer
- pip

## Run locally

From the project directory, create and activate a virtual environment, then
install the dependencies:

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python smart.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python smart.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser. The
SQLite database is created beside `smart.py` by default.

## Configuration

| Environment variable | Purpose |
| --- | --- |
| `TRAFFIC_DB` | Optional path to the SQLite database. Defaults to `traffic.db` beside `smart.py`. |
| `FLASK_SECRET_KEY` | Session signing key. Required when `APP_ENV=production`; a temporary random key is generated otherwise. |
| `APP_ENV` | Set to `production` to require an explicit session signing key. |
| `FLASK_COOKIE_SECURE` | Set to `1` when serving over HTTPS to mark session cookies as secure. Defaults to `1` in production and `0` otherwise. |

The built-in Flask server is intended for local development only. For
deployment, use a production WSGI server, configure a strong secret key and
HTTPS, and review the deployment and scaling requirements first.

## How it works

- **Road readings:** Input is validated and saved to a local SQLite database.
- **Congestion:** A simple rule uses density and speed to assign a Low, Medium,
  or High label.
- **Risk:** A Random Forest model estimates Low, Medium, or High risk from
  density, speed, weather, and visibility. Its training examples are
  synthetic.
- **Route planning:** Dijkstra's algorithm finds a minimum-cost path in a
  small, static, weighted road graph. The weights represent demo costs, not
  live travel times.
- **Signal queue:** A FIFO queue illustrates processing the next waiting
  vehicle. Queue state is in memory and resets when the app restarts.

## Project files

```text
smart.py          Flask application, dashboard, and demo algorithms
requirements.txt  Python dependencies
traffic.db        Local SQLite database, generated when the app runs
```
