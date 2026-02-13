"""
Utilidades para enviar y leer mensajes de Telegram.
"""

import os
import re
import requests
from config import TELEGRAM_API_URL


def _get_url(token):
    return TELEGRAM_API_URL.format(token=token)


def send_message(text, parse_mode='Markdown'):
    """Envia un mensaje de Telegram."""
    token = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID', '')
    if not token or not chat_id:
        print(f"[TELEGRAM] Sin credenciales. Mensaje:\n{text}")
        return False

    url = f"{_get_url(token)}/sendMessage"
    data = {
        'chat_id': chat_id,
        'text': text,
        'parse_mode': parse_mode,
    }
    try:
        resp = requests.post(url, json=data, timeout=30)
        if resp.status_code == 200:
            return True
        # Si falla Markdown, reintentar sin parse_mode
        if resp.status_code == 400 and parse_mode:
            data['parse_mode'] = None
            resp = requests.post(url, json=data, timeout=30)
            return resp.status_code == 200
        print(f"[TELEGRAM] Error {resp.status_code}: {resp.text}")
        return False
    except requests.RequestException as e:
        print(f"[TELEGRAM] Error de red: {e}")
        return False


def get_last_user_message(after_update_id=0):
    """
    Lee el ultimo mensaje del usuario en el chat.
    after_update_id: solo mensajes despues de este update_id.
    Retorna (update_id, text) o (None, None).
    """
    token = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID', '')
    if not token or not chat_id:
        return None, None

    url = f"{_get_url(token)}/getUpdates"
    params = {'offset': after_update_id + 1 if after_update_id else -10, 'limit': 100}
    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            return None, None

        data = resp.json()
        results = data.get('result', [])

        # Buscar el ultimo mensaje del usuario (no del bot) en nuestro chat
        last_update_id = None
        last_text = None
        for update in results:
            msg = update.get('message', {})
            from_user = msg.get('from', {})
            msg_chat = msg.get('chat', {})

            # Solo mensajes de nuestro chat y que no sean del bot
            if str(msg_chat.get('id')) == str(chat_id) and not from_user.get('is_bot', False):
                last_update_id = update.get('update_id')
                last_text = msg.get('text', '')

        return last_update_id, last_text

    except requests.RequestException:
        return None, None


def parse_bank_response(text):
    """
    Extrae un numero de la respuesta del usuario.
    Acepta: '500', '$500', '500.50', 'mi bank es 500', etc.
    Retorna float o None.
    """
    if not text:
        return None
    # Buscar numeros con decimales opcionales
    numbers = re.findall(r'[\d,]+\.?\d*', text.replace(',', ''))
    if numbers:
        try:
            return float(numbers[0])
        except ValueError:
            pass
    return None
