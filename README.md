# F1 Analytics Platform

## Overview

A production-grade Formula 1 analytics platform with 9 dashboard modules covering race strategy, telemetry, sentiment analysis, championship prediction, sponsorship valuation, and computer vision. Built with Streamlit and PostgreSQL, featuring real-time NLP on RSS feeds, Monte Carlo simulation, and interactive Plotly visualizations.

## Dashboard Modules

### 1. System Architecture
Displays an interactive Graphviz-rendered pipeline diagram showing the full data flow from ingestion to visualization. Includes platform KPI cards: data points ingested, prediction accuracy, uptime, and average latency.

### 2. Race Strategy Optimizer
Analyzes fuel-corrected tyre degradation curves by querying the PostgreSQL database for lap times, fuel loads, and tyre compound data. Produces Plotly charts showing mechanical grip loss after removing fuel-burn effects, with calculated degradation rates and crossover points.

### 3. Driver Telemetry Comparison
Side-by-side comparison of two drivers using sector split times, maximum speeds, lap time consistency (box plots), and a radar chart showing overall driver performance profiles across multiple dimensions.

### 4. F1 News Sentiment Analysis
Fetches live RSS feeds from multiple F1 news sources, runs TextBlob NLP sentiment analysis on each headline, and displays:
- Average sentiment score with polarity gauge
- Positive/negative headline breakdown
- Clickable headlines with sentiment indicators

### 5. Championship Simulator
Monte Carlo simulation engine that runs up to 10,000 race scenarios to predict the World Championship outcome. Configurable simulation count with real-time results visualization.

### 6. Sponsorship Value Estimator
Data-driven model for estimating team and driver sponsorship value based on performance metrics, media exposure, and historical data.

### 7. Weather Impact Analysis
Integrates weather data to analyze and predict the impact of weather conditions on race performance and strategy decisions.

### 8. Predictive Maintenance
Component reliability analytics for predicting mechanical failures and optimizing maintenance schedules.

### 9. Pit Stop Video Analyzer
Computer vision module that analyzes pit stop video footage to extract performance metrics and identify areas for improvement.

## Architecture

```
Data Sources                    Processing Layer              Presentation
+--------------------+         +---------------------+       +------------------+
| Ergast F1 API      |--JSON-->|                     |       |                  |
| (Race Data)        |         | PostgreSQL Database  |------>| Streamlit        |
+--------------------+         | (SQLAlchemy ORM)    |       | Dashboard        |
                               |                     |       | (Plotly Dark     |
+--------------------+         +---------------------+       |  Theme)          |
| RSS News Feeds     |--XML--->|                     |       |                  |
| (F1 Media)         |         | NLP Pipeline        |------>|                  |
+--------------------+         | (TextBlob)          |       |                  |
                               +---------------------+       |                  |
+--------------------+                                        |                  |
| Video Feeds        |-------->| CV Service          |------->|                  |
| (Pit Stops)        |         | (OpenCV)            |       +------------------+
+--------------------+         +---------------------+
```

## Tech Stack

- **Dashboard:** Streamlit
- **Database:** PostgreSQL (SQLAlchemy ORM)
- **Visualization:** Plotly (dark theme), Graphviz
- **NLP:** TextBlob, feedparser (RSS)
- **Simulation:** Custom Monte Carlo engine
- **Computer Vision:** OpenCV-based pit stop analysis
- **Language:** Python

## Project Structure

```
f1-analytics/
├── app.py                      # Main Streamlit application (437 lines, 9 modules)
├── cv_service.py               # Computer vision for pit stop analysis
├── sentiment_service.py        # NLP sentiment analysis pipeline
├── sponsorship_service.py      # Sponsorship value estimation model
├── strategy_optimizer.py       # Race strategy optimization engine
├── simulation.py               # Monte Carlo championship simulation
└── requirements.txt            # Python dependencies
```

## Getting Started

```bash
# Clone the repository
git clone https://github.com/ShubhGTiwari/f1-analytics.git
cd f1-analytics

# Install dependencies
pip install -r requirements.txt

# Configure database connection in Streamlit secrets
# .streamlit/secrets.toml:
# DB_CONNECTION_URI = "postgresql://user:pass@host:5432/f1_db"

# Launch the dashboard
streamlit run app.py
```

## License

MIT
