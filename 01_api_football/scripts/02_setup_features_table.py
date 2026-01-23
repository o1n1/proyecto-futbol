"""
Script para crear la tabla fixture_features en Supabase
Ejecuta el DDL via la API REST de Supabase
"""

import requests
import urllib3

# Deshabilitar warnings de SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuración
SUPABASE_URL = "https://ykqaplnfrhvjqkvejudg.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlrcWFwbG5mcmh2anFrdmVqdWRnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Njg2NjY4NjgsImV4cCI6MjA4NDI0Mjg2OH0.abeJY6QxUn4gT5GYJmoD2xJ7uPVNEwAVAxJ0wE5bMvM"

def check_table_exists():
    """Verifica si la tabla fixture_features ya existe"""
    url = f"{SUPABASE_URL}/rest/v1/fixture_features?select=fixture_id&limit=1"
    headers = {
        'apikey': SUPABASE_KEY,
        'Authorization': f'Bearer {SUPABASE_KEY}',
    }

    response = requests.get(url, headers=headers, verify=False)

    if response.status_code == 200:
        print("[OK] La tabla fixture_features ya existe")
        return True
    elif response.status_code == 404 or 'does not exist' in response.text.lower():
        print("[X] La tabla fixture_features NO existe")
        return False
    else:
        print(f"[?] Estado desconocido: {response.status_code}")
        print(response.text[:200])
        return None

def test_insert():
    """Prueba insertar un registro dummy para verificar la estructura"""
    url = f"{SUPABASE_URL}/rest/v1/fixture_features"
    headers = {
        'apikey': SUPABASE_KEY,
        'Authorization': f'Bearer {SUPABASE_KEY}',
        'Content-Type': 'application/json',
        'Prefer': 'return=minimal'
    }

    # Registro de prueba con un fixture_id que sabemos que existe
    test_data = {
        'fixture_id': 1326687,  # Primer fixture del dataset
        'home_form_points_last5': 1.5,
        'away_form_points_last5': 1.2,
        'day_of_week': 5,
        'month': 10
    }

    response = requests.post(url, headers=headers, json=test_data, verify=False)

    if response.status_code in [200, 201]:
        print("[OK] Insert de prueba exitoso")

        # Eliminar el registro de prueba
        delete_url = f"{SUPABASE_URL}/rest/v1/fixture_features?fixture_id=eq.1326687"
        requests.delete(delete_url, headers=headers, verify=False)
        print("[OK] Registro de prueba eliminado")
        return True
    else:
        print(f"[X] Error en insert: {response.status_code}")
        print(response.text[:500])
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("VERIFICACIÓN DE TABLA fixture_features")
    print("=" * 50)
    print()

    exists = check_table_exists()

    if exists:
        print("\nProbando estructura de la tabla...")
        test_insert()
    else:
        print("\n" + "=" * 50)
        print("INSTRUCCIONES PARA CREAR LA TABLA")
        print("=" * 50)
        print("""
La tabla no existe. Para crearla:

1. Ve al Dashboard de Supabase: https://supabase.com/dashboard
2. Selecciona el proyecto 'Futbol'
3. Ve a 'SQL Editor'
4. Copia y ejecuta el contenido de: create_features_table.sql

O usa el CLI de Supabase si lo tienes instalado.
        """)
