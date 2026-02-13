/**
 * Cloudflare Worker - Telegram Bot Webhook Dispatcher
 *
 * Recibe comandos de Telegram y dispara GitHub Actions workflows.
 * Comandos: /hoy, /manana, /bank <N>, /resultados, /estado, /help
 *
 * Secrets (configurar en Cloudflare dashboard o wrangler):
 * - TELEGRAM_BOT_TOKEN
 * - GITHUB_PAT
 * - TELEGRAM_CHAT_ID
 */

const GITHUB_REPO = 'o1n1/proyecto-futbol';

async function handleTelegramUpdate(update, env) {
  const msg = update.message;
  if (!msg || !msg.text) return ok();

  const chatId = String(msg.chat.id);
  const allowedChat = env.TELEGRAM_CHAT_ID;

  if (chatId !== allowedChat) return ok();

  const text = msg.text.trim();

  if (text.startsWith('/')) {
    const cmd = text.split(/\s+/)[0].toLowerCase().replace(/@\w+/, '');
    const args = text.slice(text.indexOf(' ') + 1).trim();
    const hasArgs = text.includes(' ');

    switch (cmd) {
      case '/hoy':
        await sendTelegram(env, chatId, 'Buscando partidos de copa para hoy... (30-60s)');
        await dispatchWorkflow(env, 'betting_predict.yml', { target_date: 'today' });
        break;

      case '/manana':
        await sendTelegram(env, chatId, 'Buscando partidos de copa para manana... (30-60s)');
        await dispatchWorkflow(env, 'betting_predict.yml', { target_date: 'tomorrow' });
        break;

      case '/bank':
        if (!hasArgs || isNaN(parseFloat(args))) {
          await sendTelegram(env, chatId, 'Uso: /bank 500\nEscribe el monto de tu bank actual.');
          break;
        }
        await sendTelegram(env, chatId, `Bank: $${parseFloat(args).toFixed(2)}. Calculando stakes... (30-60s)`);
        await dispatchWorkflow(env, 'betting_send.yml', { bank: args });
        break;

      case '/resultados':
        await sendTelegram(env, chatId, 'Verificando resultados... (30-60s)');
        await dispatchWorkflow(env, 'betting_results.yml', { mode: 'results' });
        break;

      case '/estado':
        await sendTelegram(env, chatId, 'Consultando estadisticas... (30-60s)');
        await dispatchWorkflow(env, 'betting_results.yml', { mode: 'stats' });
        break;

      case '/help':
      case '/start':
        await sendTelegram(env, chatId,
          '*Comandos:*\n\n' +
          '/hoy - Partidos de copa hoy\n' +
          '/manana - Partidos de copa manana\n' +
          '/bank 500 - Establece bank y envia apuestas\n' +
          '/resultados - Verifica resultados\n' +
          '/estado - Stats acumuladas (WR, ROI)\n' +
          '/help - Este mensaje'
        );
        break;

      default:
        await sendTelegram(env, chatId, 'Comando no reconocido. /help para opciones.');
    }
  } else {
    // Mensaje sin comando - interpretar como bank si es numero
    const num = parseFloat(text.replace(/[,$]/g, ''));
    if (!isNaN(num) && num > 0) {
      await sendTelegram(env, chatId, `Bank: $${num.toFixed(2)}. Calculando stakes... (30-60s)`);
      await dispatchWorkflow(env, 'betting_send.yml', { bank: String(num) });
    }
  }

  return ok();
}

async function dispatchWorkflow(env, workflow, inputs) {
  const url = `https://api.github.com/repos/${GITHUB_REPO}/actions/workflows/${workflow}/dispatches`;
  const resp = await fetch(url, {
    method: 'POST',
    headers: {
      'Authorization': `token ${env.GITHUB_PAT}`,
      'Accept': 'application/vnd.github.v3+json',
      'User-Agent': 'telegram-bot-worker',
    },
    body: JSON.stringify({ ref: 'main', inputs }),
  });

  if (resp.status !== 204) {
    const body = await resp.text();
    console.log(`GitHub dispatch error: ${resp.status} - ${body}`);
    await sendTelegram(env, env.TELEGRAM_CHAT_ID, `Error: ${resp.status}`);
  }
}

async function sendTelegram(env, chatId, text) {
  const url = `https://api.telegram.org/bot${env.TELEGRAM_BOT_TOKEN}/sendMessage`;
  const resp = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ chat_id: chatId, text, parse_mode: 'Markdown' }),
  });
  if (!resp.ok) {
    // Retry without Markdown on parse error
    await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ chat_id: chatId, text }),
    });
  }
}

function ok() {
  return new Response('ok', { status: 200 });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname === '/' || url.pathname === '/health') {
      return new Response(JSON.stringify({ status: 'ok', service: 'telegram-bot' }), {
        headers: { 'Content-Type': 'application/json' },
      });
    }

    if (url.pathname === '/webhook' && request.method === 'POST') {
      try {
        const update = await request.json();
        return await handleTelegramUpdate(update, env);
      } catch (e) {
        console.log(`Webhook error: ${e.message}`);
        return ok();
      }
    }

    return new Response('Not found', { status: 404 });
  },
};
