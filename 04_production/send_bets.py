"""
Paso 2: Leer bank del usuario, calcular stakes y enviar apuestas.
Se ejecuta diario a las 5:00pm Leon (23:00 UTC), 30 min despues de predict_cups.
Acepta variable de entorno BANK_AMOUNT para recibir bank directamente.
"""

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from telegram_utils import get_last_user_message, parse_bank_response, send_message
from config import DB_NAME, STAKE_PCT, MAX_BETS_FOR_PCT, BANK_DEFAULT

DB_PATH = Path(__file__).parent / DB_NAME


def get_pending_matches():
    """Lee partidos detectados pero no enviados."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        SELECT * FROM predictions
        WHERE status = 'detected'
        ORDER BY match_date, cup_name
    """)
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def get_last_bank():
    """Obtiene el ultimo bank registrado, o el default."""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("""
        SELECT bank_reported FROM bankroll_log
        ORDER BY id DESC LIMIT 1
    """)
    row = c.fetchone()
    conn.close()
    return row[0] if row else BANK_DEFAULT


def get_user_bank():
    """Lee el ultimo mensaje del usuario para obtener su bank."""
    # Leer ultimo update_id guardado
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("SELECT value FROM bot_state WHERE key = 'last_update_id'")
    row = c.fetchone()
    last_update_id = int(row[0]) if row else 0
    conn.close()

    update_id, text = get_last_user_message(after_update_id=last_update_id)

    if update_id and text:
        bank = parse_bank_response(text)
        if bank and bank > 0:
            # Guardar update_id para no releerlo
            conn = sqlite3.connect(str(DB_PATH))
            c = conn.cursor()
            c.execute("""
                INSERT OR REPLACE INTO bot_state (key, value)
                VALUES ('last_update_id', ?)
            """, (str(update_id),))
            conn.commit()
            conn.close()
            return bank, True

    return get_last_bank(), False


def calculate_stakes(bank, n_bets):
    """
    Calcula stake por apuesta.
    <= 10 apuestas: 10% del bank cada una
    > 10 apuestas: bank / n_bets
    """
    if n_bets <= MAX_BETS_FOR_PCT:
        return round(bank * STAKE_PCT, 2)
    else:
        return round(bank / n_bets, 2)


def update_predictions(matches, stake, bank):
    """Actualiza predicciones con stake y marca como 'pending' (apostadas)."""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    for m in matches:
        c.execute("""
            UPDATE predictions
            SET stake = ?, status = 'pending'
            WHERE id = ?
        """, (stake, m['id']))

    # Registrar bank en log
    match_date = matches[0]['match_date'] if matches else ''
    c.execute("""
        INSERT INTO bankroll_log (date, bank_reported, total_staked, bets_count)
        VALUES (?, ?, ?, ?)
    """, (match_date, bank, stake * len(matches), len(matches)))

    conn.commit()
    conn.close()


def send_bet_message(matches, stake, bank, bank_was_provided):
    """Envia mensaje con las apuestas formateadas."""
    match_date = matches[0]['match_date'] if matches else '?'
    bank_source = "reportado" if bank_was_provided else "anterior"

    lines = [f"*APUESTAS {match_date}* | Bank ({bank_source}): ${bank:,.2f}\n"]

    total_staked = 0
    for i, m in enumerate(matches, 1):
        lines.append(f"{i}. {m['home_team']} vs {m['away_team']}")
        lines.append(f"   {m['cup_name']} | Over 2.5 @ {m['odds_over25']:.2f}")
        lines.append(f"   Stake: ${stake:,.2f}")
        lines.append("")
        total_staked += stake

    pct_bank = (total_staked / bank * 100) if bank > 0 else 0
    lines.append(f"Total apostado: ${total_staked:,.2f} ({pct_bank:.0f}% del bank)")

    if not bank_was_provided:
        lines.append("\n_No respondiste. Usando ultimo bank registrado._")

    message = "\n".join(lines)
    send_message(message)


def main():
    print("=== SEND BETS ===")

    matches = get_pending_matches()
    if not matches:
        print("No hay partidos pendientes de enviar.")
        send_message("No hay partidos pendientes para apostar.")
        return

    print(f"{len(matches)} partidos pendientes.")

    # Obtener bank: primero de env (comando /bank), luego de Telegram, luego ultimo registrado
    bank_env = os.environ.get('BANK_AMOUNT', '').strip()
    if bank_env:
        try:
            bank = float(bank_env)
            was_provided = True
            print(f"Bank recibido por comando: ${bank:,.2f}")
        except ValueError:
            bank, was_provided = get_user_bank()
    else:
        bank, was_provided = get_user_bank()

    print(f"Bank: ${bank:,.2f} ({'proporcionado' if was_provided else 'ultimo registrado'})")

    # Calcular stake
    stake = calculate_stakes(bank, len(matches))
    print(f"Stake por apuesta: ${stake:,.2f} ({len(matches)} apuestas)")

    # Actualizar DB y enviar
    update_predictions(matches, stake, bank)
    send_bet_message(matches, stake, bank, was_provided)
    print("Apuestas enviadas por Telegram.")


if __name__ == '__main__':
    main()
