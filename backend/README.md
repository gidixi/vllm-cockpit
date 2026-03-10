# vLLM Backend API

Backend Node.js/Express per controllare vLLM.

## API Endpoints

- `GET /api/status` - Ottieni lo stato di vLLM e i log
- `POST /api/start` - Avvia vLLM con i parametri specificati
- `POST /api/stop` - Ferma vLLM
- `GET /health` - Health check

## Sviluppo

```bash
cd backend
npm install
npm start
```

L'API sarà disponibile su http://localhost:3001
