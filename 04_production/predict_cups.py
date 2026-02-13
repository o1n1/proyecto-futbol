"""
Paso 1: Detectar partidos de copa de manana y pedir bank al usuario.
Se ejecuta diario a las 4:30pm Leon (22:30 UTC).
"""

import sqlite3
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from sofascore_api import get_events_by_date, get_over25_odds
from telegram_utils import send_message
from config import DB_NAME

DB_PATH = Path(__file__).parent / DB_NAME

# Zona horaria de Leon GTO (UTC-6)
LEON_TZ = timezone(timedelta(hours=-6))


def get_tomorrow_date():
    """Retorna la fecha de manana en formato YYYY-MM-DD (hora Leon GTO)."""
    now_leon = datetime.now(LEON_TZ)
    tomorrow = now_leon + timedelta(days=1)
    return tomorrow.strftime('%Y-%m-%d')


def find_cup_matches(target_date):
    """Busca partidos de copa para la fecha dada y obtiene cuotas Over 2.5.

    Consulta la fecha en SofaScore (UTC) y tambien la fecha siguiente,
    porque partidos nocturnos en Europa (ej: 01:00 UTC del dia siguiente)
    son del mismo dia en hora Leon (19:00 Leon del dia anterior).
    """
    print(f"Buscando partidos de copa para {target_date} (hora Leon)...")

    # Consultar fecha target y dia siguiente en SofaScore (que usa UTC)
    next_date = (datetime.strptime(target_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    events_today = get_events_by_date(target_date) or []
    events_next = get_events_by_date(next_date) or []

    # Combinar y deduplicar por event_id
    seen_ids = set()
    all_events = []
    for event in events_today + events_next:
        eid = event.get('event_id')
        if eid not in seen_ids:
            seen_ids.add(eid)
            all_events.append(event)

    if not all_events:
        print("No se encontraron partidos de copa.")
        return []

    # Filtrar: solo partidos que son de target_date en hora Leon y que no empezaron
    upcoming = []
    for event in all_events:
        status = event.get('status_type', '')
        if status in ('finished', 'canceled', 'postponed'):
            print(f"  SKIP ({status}): {event['home_team']} vs {event['away_team']}")
            continue

        # Verificar que la fecha del partido en hora Leon corresponde a target_date
        ts = event.get('start_timestamp')
        if ts:
            event_date_leon = datetime.fromtimestamp(ts, tz=LEON_TZ).strftime('%Y-%m-%d')
            if event_date_leon != target_date:
                print(f"  SKIP (fecha Leon={event_date_leon}, target={target_date}): "
                      f"{event['home_team']} vs {event['away_team']}")
                continue

        upcoming.append(event)

    if not upcoming:
        print(f"Encontrados {len(all_events)} partidos de copa, pero ninguno aplica para {target_date} en hora Leon.")
        return []

    print(f"Encontrados {len(upcoming)} partidos de copa por jugar. Obteniendo cuotas...")

    matches = []
    for event in upcoming:
        odds = get_over25_odds(event['event_id'])
        if odds and odds.get('over_current'):
            event['odds_over25'] = odds['over_current']
            event['odds_over25_open'] = odds.get('over_open')
            event['odds_change'] = odds.get('over_change', 0)
            matches.append(event)
            print(f"  {event['cup_name']}: {event['home_team']} vs {event['away_team']} "
                  f"| Over 2.5 @ {odds['over_current']}")
        else:
            print(f"  {event['cup_name']}: {event['home_team']} vs {event['away_team']} "
                  f"| Sin cuota Over 2.5")

    return matches


def save_pending_matches(matches, target_date):
    """Guarda los partidos encontrados en bets.db como pendientes."""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    # Limpiar partidos pendientes anteriores que no se enviaron
    c.execute("DELETE FROM predictions WHERE status = 'detected' AND match_date = ?",
              (target_date,))

    for m in matches:
        c.execute("""
            INSERT OR REPLACE INTO predictions
            (match_date, event_id, cup_name, home_team, away_team, odds_over25, status)
            VALUES (?, ?, ?, ?, ?, ?, 'detected')
        """, (
            target_date,
            m['event_id'],
            m['cup_name'],
            m['home_team'],
            m['away_team'],
            m['odds_over25'],
        ))

    # Guardar timestamp para que send_bets sepa desde cuando leer mensajes
    c.execute("""
        INSERT OR REPLACE INTO bot_state (key, value)
        VALUES ('last_predict_timestamp', ?)
    """, (datetime.now(timezone.utc).isoformat(),))

    conn.commit()
    conn.close()


def send_bank_request(matches, target_date):
    """Envia mensaje Telegram pidiendo el bank actual."""
    lines = [f"*Partidos de copa para {target_date}*\n"]

    for i, m in enumerate(matches, 1):
        lines.append(f"{i}. {m['home_team']} vs {m['away_team']}")
        lines.append(f"   {m['cup_name']} | Over 2.5 @ {m['odds_over25']:.2f}")
        lines.append("")

    lines.append(f"Total: {len(matches)} apuestas")
    lines.append("")
    lines.append("Cual es tu bank actual? Responde con el numero (ej: 500)")

    message = "\n".join(lines)
    send_message(message)
    print(f"Mensaje enviado a Telegram pidiendo bank.")


def get_today_date():
    """Retorna la fecha de hoy en formato YYYY-MM-DD (hora Leon GTO)."""
    return datetime.now(LEON_TZ).strftime('%Y-%m-%d')


def get_target_date():
    """Determina fecha objetivo desde variable de entorno TARGET_DATE."""
    target = os.environ.get('TARGET_DATE', 'tomorrow').lower()
    if target == 'today':
        return get_today_date()
    else:
        return get_tomorrow_date()


def main():
    target_date = get_target_date()
    label = 'hoy' if os.environ.get('TARGET_DATE', '').lower() == 'today' else 'manana'
    print(f"=== PREDICT CUPS - {target_date} ({label}) ===")

    matches = find_cup_matches(target_date)

    if not matches:
        send_message(f"Sin apuestas de copa para {label} {target_date}")
        print("No hay partidos. Mensaje enviado.")
        return

    save_pending_matches(matches, target_date)
    send_bank_request(matches, target_date)
    print(f"Listo. {len(matches)} partidos guardados. Esperando respuesta del usuario.")


if __name__ == '__main__':
    main()
