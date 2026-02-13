"""
Inicializa la base de datos bets.db con las tablas necesarias.
Ejecutar una sola vez antes de poner en produccion.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / 'bets.db'


def init_db():
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT DEFAULT (datetime('now')),
            match_date TEXT NOT NULL,
            event_id INTEGER NOT NULL,
            cup_name TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            odds_over25 REAL,
            stake REAL,
            status TEXT DEFAULT 'pending',
            home_score INTEGER,
            away_score INTEGER,
            total_goals INTEGER,
            pnl REAL,
            settled_at TEXT,
            UNIQUE(event_id, match_date)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS bankroll_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            bank_reported REAL,
            total_staked REAL,
            total_pnl REAL,
            bets_count INTEGER,
            bets_won INTEGER,
            bets_lost INTEGER
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS bot_state (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    conn.commit()
    conn.close()
    print(f"Base de datos inicializada en: {DB_PATH}")


if __name__ == '__main__':
    init_db()
