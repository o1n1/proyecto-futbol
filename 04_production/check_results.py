"""
Paso 3: Verificar resultados de partidos apostados y actualizar P&L.
Se ejecuta diario a las 2:00am Leon (08:00 UTC).
Acepta variable de entorno MODE: 'results' (default) o 'stats' (solo estadisticas).
"""

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from sofascore_api import get_event_result, get_over25_odds
from telegram_utils import send_message
from config import DB_NAME

DB_PATH = Path(__file__).parent / DB_NAME


def get_pending_predictions():
    """Lee predicciones con status='pending' (apostadas, sin resultado)."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        SELECT * FROM predictions
        WHERE status = 'pending'
        ORDER BY match_date, cup_name
    """)
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def settle_prediction(pred):
    """
    Verifica resultado de un partido y calcula P&L.
    Retorna dict actualizado o None si no ha terminado.
    """
    result = get_event_result(pred['event_id'])
    if not result:
        return None

    if not result['is_finished']:
        return None

    total_goals = result['total_goals']
    if total_goals is None:
        return None

    over25_won = total_goals > 2.5
    odds = pred['odds_over25'] or 1.0
    stake = pred['stake'] or 0

    if over25_won:
        pnl = round(stake * (odds - 1), 2)
        status = 'won'
    else:
        pnl = round(-stake, 2)
        status = 'lost'

    return {
        'id': pred['id'],
        'home_score': result['home_score'],
        'away_score': result['away_score'],
        'total_goals': total_goals,
        'status': status,
        'pnl': pnl,
        'settled_at': datetime.now(timezone.utc).isoformat(),
    }


def update_settled(settled_list):
    """Actualiza predicciones resueltas en la DB."""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    for s in settled_list:
        c.execute("""
            UPDATE predictions
            SET home_score = ?, away_score = ?, total_goals = ?,
                status = ?, pnl = ?, settled_at = ?
            WHERE id = ?
        """, (
            s['home_score'], s['away_score'], s['total_goals'],
            s['status'], s['pnl'], s['settled_at'], s['id']
        ))

    # Actualizar bankroll_log del dia si existe
    if settled_list:
        won = sum(1 for s in settled_list if s['status'] == 'won')
        lost = sum(1 for s in settled_list if s['status'] == 'lost')
        total_pnl = sum(s['pnl'] for s in settled_list)

        # Buscar el log mas reciente y actualizarlo
        c.execute("""
            UPDATE bankroll_log
            SET total_pnl = ?, bets_won = ?, bets_lost = ?
            WHERE id = (SELECT MAX(id) FROM bankroll_log)
        """, (total_pnl, won, lost))

    conn.commit()
    conn.close()


def get_accumulated_stats():
    """Obtiene estadisticas acumuladas de todas las apuestas."""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    c.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN status = 'won' THEN 1 ELSE 0 END) as won,
            SUM(CASE WHEN status = 'lost' THEN 1 ELSE 0 END) as lost,
            SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
            SUM(CASE WHEN status IN ('won', 'lost') THEN pnl ELSE 0 END) as total_pnl,
            SUM(CASE WHEN status IN ('won', 'lost') THEN stake ELSE 0 END) as total_staked
        FROM predictions
    """)
    row = c.fetchone()
    conn.close()

    total, won, lost, pending, total_pnl, total_staked = row
    total_pnl = total_pnl or 0
    total_staked = total_staked or 0
    settled = (won or 0) + (lost or 0)
    wr = (won / settled * 100) if settled > 0 else 0
    roi = (total_pnl / total_staked * 100) if total_staked > 0 else 0

    return {
        'total': total or 0,
        'won': won or 0,
        'lost': lost or 0,
        'pending': pending or 0,
        'total_pnl': total_pnl,
        'total_staked': total_staked,
        'win_rate': wr,
        'roi': roi,
    }


def send_results_message(settled_list, stats):
    """Envia resumen de resultados por Telegram."""
    if not settled_list:
        # Verificar si hay pendientes
        if stats['pending'] > 0:
            send_message(f"Aun hay {stats['pending']} apuestas pendientes de resultado.")
        return

    # Agrupar por fecha
    dates = set(s.get('match_date', '?') for s in settled_list)
    date_str = ', '.join(sorted(dates)) if dates else '?'

    lines = [f"*RESULTADOS*\n"]

    day_pnl = 0
    day_won = 0
    day_lost = 0
    for s in settled_list:
        emoji = "WIN" if s['status'] == 'won' else "LOSS"
        score = f"{s['home_score']}-{s['away_score']}"
        pnl_str = f"+${s['pnl']:,.2f}" if s['pnl'] >= 0 else f"-${abs(s['pnl']):,.2f}"
        # Buscar nombre del equipo en la prediccion original
        lines.append(f"{emoji} {s.get('home_team', '?')} {score} {s.get('away_team', '?')}")
        lines.append(f"   {s.get('cup_name', '?')} | Over 2.5 @ {s.get('odds', '?'):.2f} | {pnl_str}")
        lines.append("")
        day_pnl += s['pnl']
        if s['status'] == 'won':
            day_won += 1
        else:
            day_lost += 1

    total_day = day_won + day_lost
    wr_day = (day_won / total_day * 100) if total_day > 0 else 0
    pnl_str = f"+${day_pnl:,.2f}" if day_pnl >= 0 else f"-${abs(day_pnl):,.2f}"

    lines.append(f"Dia: {day_won}/{total_day} ({wr_day:.0f}%) | P&L: {pnl_str}")
    lines.append("")
    lines.append(f"*Acumulado*: {stats['won']}/{stats['won']+stats['lost']} "
                 f"({stats['win_rate']:.1f}% WR) | ROI: {stats['roi']:+.1f}%")
    lines.append(f"P&L total: ${stats['total_pnl']:+,.2f}")

    if stats['pending'] > 0:
        lines.append(f"\n_{stats['pending']} apuestas aun pendientes_")

    message = "\n".join(lines)
    send_message(message)


def send_stats_message():
    """Envia solo las estadisticas acumuladas (comando /estado)."""
    stats = get_accumulated_stats()
    if stats['total'] == 0:
        send_message("No hay apuestas registradas aun.")
        return

    settled = stats['won'] + stats['lost']
    lines = ["*ESTADISTICAS ACUMULADAS*\n"]
    lines.append(f"Total apuestas: {stats['total']}")
    lines.append(f"Resueltas: {settled} ({stats['won']}W / {stats['lost']}L)")
    if stats['pending'] > 0:
        lines.append(f"Pendientes: {stats['pending']}")
    lines.append("")
    lines.append(f"Win Rate: {stats['win_rate']:.1f}%")
    lines.append(f"ROI: {stats['roi']:+.1f}%")
    lines.append(f"P&L total: ${stats['total_pnl']:+,.2f}")
    lines.append(f"Total apostado: ${stats['total_staked']:,.2f}")

    send_message("\n".join(lines))


def main():
    mode = os.environ.get('MODE', 'results').lower()
    print(f"=== CHECK RESULTS (mode={mode}) ===")

    if mode == 'stats':
        send_stats_message()
        print("Estadisticas enviadas.")
        return

    predictions = get_pending_predictions()
    if not predictions:
        print("No hay predicciones pendientes.")
        stats = get_accumulated_stats()
        if stats['total'] > 0:
            send_message(f"Sin apuestas pendientes.\n"
                         f"Acumulado: {stats['won']}/{stats['won']+stats['lost']} "
                         f"({stats['win_rate']:.1f}% WR) | ROI: {stats['roi']:+.1f}%")
        else:
            send_message("No hay apuestas registradas aun.")
        return

    print(f"{len(predictions)} predicciones pendientes. Verificando resultados...")

    settled_list = []
    still_pending = 0
    for pred in predictions:
        result = settle_prediction(pred)
        if result:
            result['home_team'] = pred['home_team']
            result['away_team'] = pred['away_team']
            result['cup_name'] = pred['cup_name']
            result['odds'] = pred['odds_over25']
            result['match_date'] = pred['match_date']
            settled_list.append(result)
            print(f"  {result['status'].upper()}: {pred['home_team']} "
                  f"{result['home_score']}-{result['away_score']} {pred['away_team']} "
                  f"| PnL: ${result['pnl']:+.2f}")
        else:
            still_pending += 1
            print(f"  PENDING: {pred['home_team']} vs {pred['away_team']}")

    if settled_list:
        update_settled(settled_list)

    stats = get_accumulated_stats()
    send_results_message(settled_list, stats)

    print(f"\nResueltos: {len(settled_list)} | Pendientes: {still_pending}")
    print(f"Acumulado: {stats['won']}/{stats['won']+stats['lost']} "
          f"({stats['win_rate']:.1f}% WR) | ROI: {stats['roi']:+.1f}%")


if __name__ == '__main__':
    main()
