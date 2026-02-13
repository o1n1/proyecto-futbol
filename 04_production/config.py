"""
Configuracion del sistema de apuestas automatizado.
Over 2.5 en copas europeas.
"""

# 18 copas europeas validadas en backtesting (~75% WR, ~22% ROI)
COPAS_EUROPEAS = [
    'FA Cup', 'FA Cup, Qualification', 'EFL Cup',
    'Copa del Rey', 'Coupe de France',
    'KNVB beker', 'Beker van Belgie',
    'DFB Pokal', 'Coppa Italia',
    'Turkiye Kupasi, Qualification',
    'Challenge Cup', 'Challenge Cup, Playoffs',
    'NM Cup', 'Czech Cup', 'OFB Cup',
    'Oddset Pokalen', 'Schweizer Cup',
    'Scottish League Cup',
]

# Aliases: nombres que SofaScore puede usar vs nuestros nombres
COPA_ALIASES = {
    'Belgian Cup': 'Beker van Belgie',
    'League Cup': 'EFL Cup',
    'Carabao Cup': 'EFL Cup',
    'Turkish Cup': 'Turkiye Kupasi, Qualification',
    'NM Cupen': 'NM Cup',
    'Swiss Cup': 'Schweizer Cup',
    'Scottish Challenge Cup': 'Challenge Cup',
    'Scottish Challenge Cup, Playoffs': 'Challenge Cup, Playoffs',
}

# Stake
STAKE_PCT = 0.10        # 10% del bank por apuesta
MAX_BETS_FOR_PCT = 10   # Si >10 apuestas: dividir bank entre todas
BANK_DEFAULT = 500      # Bank por defecto si usuario no responde

# SofaScore API
SOFASCORE_BASE_URL = 'https://api.sofascore.com/api/v1'
SOFASCORE_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.sofascore.com/',
    'Origin': 'https://www.sofascore.com',
}

# Telegram
TELEGRAM_API_URL = 'https://api.telegram.org/bot{token}'

# DB
DB_NAME = 'bets.db'
