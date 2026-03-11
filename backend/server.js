const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const { spawn, exec } = require('child_process');
const fs = require('fs');
const path = require('path');
const { promisify } = require('util');

const execAsync = promisify(exec);

const app = express();
const PORT = 3001;

app.use(cors());
app.use(bodyParser.json());

let vllmProcess = null;
let vllmLogs = [];
let vllmStatus = 'stopped';

// Funzione helper per contare le GPU disponibili
async function getAvailableGpuCount() {
  try {
    const { stdout } = await execAsync('nvidia-smi --list-gpus | wc -l', { timeout: 5000 });
    const count = parseInt(stdout.trim(), 10);
    return isNaN(count) ? 0 : count;
  } catch (error) {
    // Se nvidia-smi non è disponibile, assumi 0 GPU
    return 0;
  }
}

// Funzione per avviare vLLM
async function startVLLM(config) {
  if (vllmProcess) {
    return { error: 'vLLM è già in esecuzione' };
  }

  // Valida il numero di GPU disponibili per tensor parallelism
  if (config.tensorParallelSize && config.tensorParallelSize > 1) {
    const availableGpus = await getAvailableGpuCount();
    if (availableGpus === 0) {
      return { error: 'Nessuna GPU disponibile. Verifica che nvidia-smi funzioni correttamente.' };
    }
    if (config.tensorParallelSize > availableGpus) {
      return { 
        error: `Tensor Parallel Size (${config.tensorParallelSize}) è maggiore del numero di GPU disponibili (${availableGpus}). Riduci il valore o usa --distributed-executor-backend ray per distribuire su più nodi.` 
      };
    }
  }

  vllmStatus = 'starting';
  vllmLogs = [];

  // Costruisci il comando vLLM
  // Nota: il positional è model_tag, ma l'engine usa args.model (default Qwen se non passato).
  // Passiamo esplicitamente --model per evitare che venga usato il default.
  // Host: in Docker vLLM deve ascoltare su 0.0.0.0, altrimenti la porta 8000 non è raggiungibile dal port mapping.
  const listenHost = (config.host && config.host.trim() !== '' && config.host !== '127.0.0.1') ? config.host.trim() : '0.0.0.0';
  const args = [
    '-m', 'vllm.entrypoints.openai.api_server',
    config.model,
    '--model', config.model,
    '--host', listenHost,
    '--port', config.port.toString(),
  ];

  // Parallelismo
  if (config.tensorParallelSize && config.tensorParallelSize > 1) {
    args.push('--tensor-parallel-size', config.tensorParallelSize.toString());
  }
  if (config.pipelineParallelSize && config.pipelineParallelSize > 1) {
    args.push('--pipeline-parallel-size', config.pipelineParallelSize.toString());
  }
  if (config.dataParallelSize && config.dataParallelSize > 1) {
    args.push('--data-parallel-size', config.dataParallelSize.toString());
  }

  // Memoria GPU
  if (config.gpuMemoryUtilization !== undefined) {
    args.push('--gpu-memory-utilization', config.gpuMemoryUtilization.toString());
  }
  if (config.cpuOffloadGb && config.cpuOffloadGb > 0) {
    args.push('--cpu-offload-gb', config.cpuOffloadGb.toString());
  }
  // Alcuni modelli (es. TinyLlama) hanno max 2048; se la config ha 16384 vLLM rifiuta. Adeguiamo.
  let maxModelLen = config.maxModelLen;
  if (config.model && maxModelLen) {
    const smallContextModels = ['TinyLlama', 'Phi-2', 'phi-2', '1.1B-Chat', '0.6B'];
    if (smallContextModels.some(m => config.model.includes(m))) {
      maxModelLen = Math.min(Number(maxModelLen) || 4096, 2048);
    }
  }
  if (maxModelLen) {
    args.push('--max-model-len', maxModelLen.toString());
  }
  if (config.kvCacheDtype && config.kvCacheDtype !== 'auto') {
    args.push('--kv-cache-dtype', config.kvCacheDtype);
  }
  if (config.blockSize && config.blockSize !== 16) {
    args.push('--block-size', config.blockSize.toString());
  }
  if (config.swapSpace && config.swapSpace !== 4) {
    args.push('--swap-space', config.swapSpace.toString());
  }

  // Tipo di dato e quantizzazione
  // gpt-oss (MXFP4) richiede bfloat16; se l'utente ha scelto float16/half, forziamo bfloat16
  let dtype = config.dtype;
  if (config.model && config.model.includes('gpt-oss') && (dtype === 'half' || dtype === 'float16')) {
    dtype = 'bfloat16';
  }
  if (dtype && dtype !== 'auto') {
    args.push('--dtype', dtype);
  }
  if (config.quantization && config.quantization.trim() !== '') {
    args.push('--quantization', config.quantization);
  }
  if (config.loadFormat && config.loadFormat !== 'auto') {
    args.push('--load-format', config.loadFormat);
  }
  if (config.quantizationParamPath && config.quantizationParamPath.trim() !== '') {
    args.push('--quantization-param-path', config.quantizationParamPath);
  }

  // GPU Selection
  if (config.device && config.device !== 'auto') {
    args.push('--device', config.device);
  }
  if (config.distributedExecutorBackend && config.distributedExecutorBackend !== 'ray') {
    args.push('--distributed-executor-backend', config.distributedExecutorBackend);
  }
  if (config.workerUseRay) {
    args.push('--worker-use-ray');
  }

  // KV Cache e throughput
  if (config.maxNumSeqs) {
    args.push('--max-num-seqs', config.maxNumSeqs.toString());
  }
  if (config.maxNumBatchedTokens) {
    args.push('--max-num-batched-tokens', config.maxNumBatchedTokens.toString());
  }
  if (config.enablePrefixCaching) {
    args.push('--enable-prefix-caching');
  }
  if (config.numGpuBlocksOverride && config.numGpuBlocksOverride.trim() !== '') {
    args.push('--num-gpu-blocks-override', config.numGpuBlocksOverride);
  }

  // Trust remote code
  if (config.trustRemoteCode) {
    args.push('--trust-remote-code');
  }

  // Tool calling support (required for structured tool calls via OpenAI API)
  // Auto-detect: enable by default for models known to support Hermes-style tool calling
  const isQwen = config.model && /qwen/i.test(config.model);
  const autoToolChoice = config.enableAutoToolChoice !== undefined
    ? config.enableAutoToolChoice
    : isQwen;  // Default: on for Qwen, off for others
  const toolCallParser = config.toolCallParser && config.toolCallParser.trim() !== ''
    ? config.toolCallParser.trim()
    : (isQwen ? 'hermes' : '');

  if (autoToolChoice) {
    args.push('--enable-auto-tool-choice');
    if (toolCallParser) {
      args.push('--tool-call-parser', toolCallParser);
    }
  }

  // Served model name
  if (config.servedModelName && config.servedModelName.trim() !== '') {
    args.push('--served-model-name', config.servedModelName.trim());
  }

  // Multi-modal encoder tensor parallelism mode
  if (config.mmEncoderTpMode && config.mmEncoderTpMode.trim() !== '') {
    args.push('--mm-encoder-tp-mode', config.mmEncoderTpMode.trim());
  }

  // Multi-modal processor cache type
  if (config.mmProcessorCacheType && config.mmProcessorCacheType.trim() !== '') {
    args.push('--mm-processor-cache-type', config.mmProcessorCacheType.trim());
  }

  // Reasoning parser
  if (config.reasoningParser && config.reasoningParser.trim() !== '') {
    args.push('--reasoning-parser', config.reasoningParser.trim());
  }

  // Più log: così si capisce caricamento, download, ecc.
  args.push('--uvicorn-log-level', 'info');

  // Usa Python 3.11 del sistema e imposta PYTHONPATH al venv montato
  // Il venv montato ha symlink a pyenv che non esistono nel container,
  // quindi usiamo il Python del container con i pacchetti del venv
  const pythonCmd = 'python3.11';
  
  // Prepara variabili d'ambiente
  // HF_HOME sulla cache montata: con user 1000:1000 la HOME non è /root, quindi senza questo
  // il container riscaricherebbe il modello invece di usare quello già scaricato in ~/.cache/huggingface
  const env = {
    ...process.env,
    PATH: '/usr/bin:' + process.env.PATH,
    PYTHONPATH: '/app/venv/lib/python3.11/site-packages',
    VIRTUAL_ENV: '/app/venv',
    HF_HOME: '/mnt/huggingface-gds',
    // Riduce frammentazione VRAM (suggerito da PyTorch in caso di OOM con "reserved but unallocated" grande)
    PYTORCH_ALLOC_CONF: 'expandable_segments:True',
    // Più log vLLM e Hugging Face (download/cache)
    VLLM_LOGGING_LEVEL: process.env.VLLM_LOGGING_LEVEL || 'INFO',
    HF_HUB_VERBOSITY: process.env.HF_HUB_VERBOSITY || 'info',
  };

  // Aggiungi CUDA_VISIBLE_DEVICES se specificato
  if (config.cudaVisibleDevices && config.cudaVisibleDevices.trim() !== '') {
    env.CUDA_VISIBLE_DEVICES = config.cudaVisibleDevices.trim();
  }
  
  vllmProcess = spawn(pythonCmd, args, {
    cwd: '/app',
    env: env
  });

  // Gestisci output
  vllmProcess.stdout.on('data', (data) => {
    const log = data.toString();
    vllmLogs.push(`[STDOUT] ${log.trim()}`);
    console.log('[vLLM]', log);
    // Limita i log a 1000 righe
    if (vllmLogs.length > 1000) {
      vllmLogs.shift();
    }
  });

  vllmProcess.stderr.on('data', (data) => {
    const log = data.toString();
    vllmLogs.push(`[STDERR] ${log.trim()}`);
    console.error('[vLLM]', log);
    if (vllmLogs.length > 1000) {
      vllmLogs.shift();
    }
  });

  vllmProcess.on('close', (code) => {
    vllmLogs.push(`[SYSTEM] Processo terminato con codice ${code}`);
    vllmProcess = null;
    vllmStatus = 'stopped';
    console.log(`vLLM processo terminato con codice ${code}`);
  });

  vllmProcess.on('error', (error) => {
    vllmLogs.push(`[ERROR] ${error.message}`);
    vllmStatus = 'stopped';
    vllmProcess = null;
    console.error('Errore avvio vLLM:', error);
  });

  // Attendi un po' per vedere se parte correttamente
  setTimeout(() => {
    if (vllmProcess && vllmProcess.pid) {
      vllmStatus = 'running';
    }
  }, 3000);

  return { success: true, message: 'vLLM avviato' };
}

// Funzione per fermare vLLM
function stopVLLM() {
  if (!vllmProcess) {
    return { error: 'vLLM non è in esecuzione' };
  }

  try {
    vllmProcess.kill('SIGTERM');
    vllmLogs.push('[SYSTEM] Invio segnale SIGTERM a vLLM');
    
    // Se non si ferma in 5 secondi, forza la terminazione
    setTimeout(() => {
      if (vllmProcess) {
        vllmProcess.kill('SIGKILL');
        vllmLogs.push('[SYSTEM] Forzata terminazione vLLM');
      }
    }, 5000);

    vllmStatus = 'stopping';
    return { success: true, message: 'Comando di stop inviato' };
  } catch (error) {
    return { error: error.message };
  }
}

// API Routes

// Status
app.get('/api/status', (req, res) => {
  res.json({
    status: vllmStatus,
    logs: vllmLogs.slice(-50) // Ultimi 50 log
  });
});

// Diagnostica porta 8000: verifica se vLLM risponde (da dentro il container)
app.get('/api/check-port-8000', async (req, res) => {
  try {
    const { stdout } = await execAsync(
      'curl -s -o /dev/null -w "%{http_code}" --connect-timeout 3 http://127.0.0.1:8000/v1/models',
      { timeout: 5000 }
    );
    const code = (stdout || '').trim();
    const ok = code === '200';
    res.json({
      port8000Reachable: ok,
      httpCode: code || 'timeout/error',
      hint: ok ? 'La porta 8000 risponde. Da host prova: curl http://localhost:8000/v1/models' : 'vLLM non risponde su 8000. Hai avviato il modello dalla Web UI?'
    });
  } catch (e) {
    res.json({
      port8000Reachable: false,
      httpCode: 'error',
      hint: 'vLLM non in ascolto su 8000. Avvia il modello dalla Web UI (Start). Se già avviato, controlla i log in /api/status.',
      error: e.message
    });
  }
});

// Start vLLM
app.post('/api/start', async (req, res) => {
  const config = req.body;
  
  // Valori di default
  const defaultConfig = {
    model: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    host: '0.0.0.0',
    port: 8000,
    trustRemoteCode: true,
    tensorParallelSize: 1,
    pipelineParallelSize: 1,
    dataParallelSize: 1,
    gpuMemoryUtilization: 0.9,
    cpuOffloadGb: 0,
    maxModelLen: 4096,
    kvCacheDtype: 'auto',
    blockSize: 16,
    swapSpace: 4,
    dtype: 'half',
    quantization: '',
    loadFormat: 'auto',
    quantizationParamPath: '',
    device: 'auto',
    distributedExecutorBackend: 'ray',
    workerUseRay: false,
    cudaVisibleDevices: '',
    maxNumSeqs: 256,
    maxNumBatchedTokens: 16384,
    enablePrefixCaching: false,
    numGpuBlocksOverride: '',
    // Tool calling: auto-detected per model in startVLLM(); leave undefined to use auto-detect
    // enableAutoToolChoice: undefined,
    // toolCallParser: '',
    servedModelName: '',
    mmEncoderTpMode: '',
    mmProcessorCacheType: '',
    reasoningParser: '',
  };

  const finalConfig = { ...defaultConfig, ...config };
  
  try {
    const result = await startVLLM(finalConfig);
    
    if (result.error) {
      res.status(400).json(result);
    } else {
      res.json(result);
    }
  } catch (error) {
    res.status(500).json({ error: error.message || 'Errore interno del server' });
  }
});

// Stop vLLM
app.post('/api/stop', (req, res) => {
  const result = stopVLLM();
  
  if (result.error) {
    res.status(400).json(result);
  } else {
    res.json(result);
  }
});

// Salva configurazione su file
app.post('/api/config/save', (req, res) => {
  try {
    const config = req.body.config;
    const filename = req.body.filename || `vllm-config-${Date.now()}.json`;
    const configDir = '/app/configs';
    
    // Crea directory se non esiste
    if (!fs.existsSync(configDir)) {
      fs.mkdirSync(configDir, { recursive: true });
    }
    
    const filepath = path.join(configDir, filename);
    // Salva solo la configurazione, non l'oggetto wrapper
    fs.writeFileSync(filepath, JSON.stringify(config, null, 2));
    
    res.json({ success: true, message: 'Configurazione salvata', filename, filepath });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Lista configurazioni salvate
app.get('/api/config/list', (req, res) => {
  try {
    const configDir = '/app/configs';
    
    if (!fs.existsSync(configDir)) {
      return res.json({ configs: [] });
    }
    
    const files = fs.readdirSync(configDir)
      .filter(file => file.endsWith('.json'))
      .map(file => {
        const filepath = path.join(configDir, file);
        const stats = fs.statSync(filepath);
        return {
          filename: file,
          size: stats.size,
          modified: stats.mtime.toISOString()
        };
      })
      .sort((a, b) => new Date(b.modified) - new Date(a.modified));
    
    res.json({ configs: files });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Carica configurazione da file
app.get('/api/config/load/:filename', (req, res) => {
  try {
    const filename = req.params.filename;
    const configDir = '/app/configs';
    const filepath = path.join(configDir, filename);
    
    if (!fs.existsSync(filepath)) {
      return res.status(404).json({ error: 'File non trovato' });
    }
    
    const fileContent = fs.readFileSync(filepath, 'utf8');
    const parsed = JSON.parse(fileContent);
    
    // Se il file ha una struttura { config: {...}, filename: "..." }, estrai solo config
    // Altrimenti usa l'intero oggetto come configurazione
    const config = parsed.config || parsed;
    
    res.json({ success: true, config });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Elimina configurazione
app.delete('/api/config/delete/:filename', (req, res) => {
  try {
    const filename = req.params.filename;
    const configDir = '/app/configs';
    const filepath = path.join(configDir, filename);
    
    if (!fs.existsSync(filepath)) {
      return res.status(404).json({ error: 'File non trovato' });
    }
    
    fs.unlinkSync(filepath);
    res.json({ success: true, message: 'Configurazione eliminata' });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get GPU info
app.get('/api/gpus', async (req, res) => {
  try {
    // Prova a eseguire nvidia-smi con timeout di 5 secondi
    const { stdout } = await Promise.race([
      execAsync('nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits'),
      new Promise((_, reject) => setTimeout(() => reject(new Error('Timeout')), 5000))
    ]);
    
    if (!stdout || stdout.trim().length === 0) {
      throw new Error('Nessun output da nvidia-smi');
    }
    
    const gpus = stdout.trim().split('\n')
      .filter(line => line.trim().length > 0)
      .map((line, index) => {
        const parts = line.split(', ').map(p => p.trim());
        return {
          index: parseInt(parts[0]) || index,
          name: parts[1] || 'Unknown',
          memoryTotal: parseInt(parts[2]) || 0,
          memoryUsed: parseInt(parts[3]) || 0,
          memoryFree: parseInt(parts[4]) || 0,
          utilization: parseInt(parts[5]) || 0,
          temperature: parseInt(parts[6]) || 0,
        };
      });

    res.json({ gpus, available: true });
  } catch (error) {
    // Se nvidia-smi non è disponibile, restituisci info vuota
    console.error('Errore nel recupero info GPU:', error.message);
    res.json({ 
      gpus: [], 
      available: false,
      error: error.message || 'nvidia-smi non disponibile o nessuna GPU rilevata'
    });
  }
});

// Lista modelli disponibili nella cache HuggingFace
app.get('/api/models/list', async (req, res) => {
  try {
    // Prova diverse posizioni comuni per la cache HuggingFace
    const possibleCacheDirs = [
      process.env.HF_HOME,
      process.env.HUGGINGFACE_HUB_CACHE,
      '/mnt/huggingface-gds', // Mount della cache dell'utente gds nel container
      '/home/gds/.cache/huggingface',
      '/root/.cache/huggingface',
      path.join(process.env.HOME || '/root', '.cache/huggingface')
    ].filter(Boolean); // Rimuove undefined/null
    
    let allModels = [];
    const scannedDirs = new Set();
    
    // Cerca modelli in tutte le directory possibili
    for (const cacheDir of possibleCacheDirs) {
      if (!fs.existsSync(cacheDir)) {
        console.log(`[Models] Directory non esiste: ${cacheDir}`);
        continue;
      }
      
      console.log(`[Models] Verificando directory: ${cacheDir}`);
      
      // Prova prima la nuova struttura (modelli direttamente nella cache)
      const modelsDir = cacheDir;
      if (fs.existsSync(modelsDir) && !scannedDirs.has(modelsDir)) {
        console.log(`[Models] Scansionando directory: ${modelsDir}`);
        const models = scanModelsDirectory(modelsDir);
        console.log(`[Models] Trovati ${models.length} modelli in ${modelsDir}`);
        allModels = allModels.concat(models);
        scannedDirs.add(modelsDir);
      }
      
      // Prova anche la vecchia struttura con hub/
      const hubDir = path.join(cacheDir, 'hub');
      if (fs.existsSync(hubDir) && !scannedDirs.has(hubDir)) {
        console.log(`[Models] Scansionando directory hub: ${hubDir}`);
        const models = scanModelsDirectory(hubDir);
        console.log(`[Models] Trovati ${models.length} modelli in ${hubDir}`);
        allModels = allModels.concat(models);
        scannedDirs.add(hubDir);
      }
    }
    
    // Rimuovi duplicati (stesso nome modello)
    const uniqueModels = [];
    const seenNames = new Set();
    for (const model of allModels) {
      if (!seenNames.has(model.name)) {
        seenNames.add(model.name);
        uniqueModels.push(model);
      }
    }
    
    // Ordina per data di modifica (più recenti prima)
    uniqueModels.sort((a, b) => new Date(b.modified) - new Date(a.modified));
    
    console.log(`[Models] Totale modelli unici trovati: ${uniqueModels.length}`);
    
    res.json({ models: uniqueModels, count: uniqueModels.length });
  } catch (error) {
    console.error('Errore nel recupero modelli:', error);
    res.status(500).json({ error: error.message });
  }
});

// Funzione helper per scansionare directory modelli
function scanModelsDirectory(hubDir) {
  const models = [];
  let entries;
  
  try {
    entries = fs.readdirSync(hubDir, { withFileTypes: true });
  } catch (err) {
    console.error(`[Models] Errore lettura directory ${hubDir}:`, err.message);
    return [];
  }
  
  console.log(`[Models] Trovate ${entries.length} entry in ${hubDir}`);
  
  for (const entry of entries) {
    if (entry.isDirectory() && entry.name.startsWith('models--')) {
      // Converti models--owner--model-name in owner/model-name
      const modelName = entry.name.replace(/^models--/, '').replace(/--/g, '/');
      const modelPath = path.join(hubDir, entry.name);
      
      console.log(`[Models] Processando modello: ${modelName} (${modelPath})`);
      
      try {
        // Cerca file del modello nelle directory snapshots o blobs
        const snapshotsDir = path.join(modelPath, 'snapshots');
        const blobsDir = path.join(modelPath, 'blobs');
        
        let modelFiles = [];
        let actualModelPath = modelPath;
        let hasModelFiles = false;
        
        // Cerca in snapshots (struttura più comune)
        if (fs.existsSync(snapshotsDir)) {
          console.log(`[Models] Snapshots dir trovata per ${modelName}`);
          let snapshots;
          try {
            snapshots = fs.readdirSync(snapshotsDir, { withFileTypes: true })
              .filter(e => e.isDirectory());
          } catch (err) {
            console.error(`[Models] Errore lettura snapshots per ${modelName}:`, err.message);
            snapshots = [];
          }
          
          if (snapshots.length > 0) {
            console.log(`[Models] Trovati ${snapshots.length} snapshot per ${modelName}`);
            // Prendi l'ultimo snapshot (più recente)
            const latestSnapshot = snapshots.sort((a, b) => {
              try {
                const aPath = path.join(snapshotsDir, a.name);
                const bPath = path.join(snapshotsDir, b.name);
                return fs.statSync(bPath).mtime - fs.statSync(aPath).mtime;
              } catch (err) {
                return 0;
              }
            })[0];
            
            actualModelPath = path.join(snapshotsDir, latestSnapshot.name);
            console.log(`[Models] Usando snapshot: ${actualModelPath}`);
            
            let allEntries;
            try {
              allEntries = fs.readdirSync(actualModelPath, { withFileTypes: true });
            } catch (err) {
              console.error(`[Models] Errore lettura snapshot ${actualModelPath}:`, err.message);
              allEntries = [];
            }
            
            // Conta sia file che symlink (i modelli HuggingFace usano symlink)
            modelFiles = allEntries
              .filter(e => e.isFile() || e.isSymbolicLink())
              .map(e => e.name);
            
            console.log(`[Models] File trovati in snapshot:`, modelFiles.slice(0, 5));
            
            hasModelFiles = modelFiles.some(f => 
              f.endsWith('.bin') || 
              f.endsWith('.safetensors') || 
              f.endsWith('.pt') ||
              f.endsWith('.gguf') ||
              f === 'config.json' ||
              f === 'model_index.json' ||
              f === 'tokenizer.json' ||
              f.startsWith('model.')
            );
            
            console.log(`[Models] Modello ${modelName}: ${modelFiles.length} file, hasModelFiles=${hasModelFiles}`);
          }
        }
        
        // Se non trovato in snapshots, cerca in blobs
        if (!hasModelFiles && fs.existsSync(blobsDir)) {
          console.log(`[Models] Cercando in blobs per ${modelName}`);
          try {
            modelFiles = fs.readdirSync(blobsDir);
            hasModelFiles = modelFiles.length > 0;
            actualModelPath = blobsDir;
          } catch (err) {
            console.error(`[Models] Errore lettura blobs per ${modelName}:`, err.message);
          }
        }
        
        if (hasModelFiles) {
          console.log(`[Models] Aggiungendo modello ${modelName} alla lista`);
          const stats = fs.statSync(modelPath);
          const size = getDirectorySize(modelPath);
          models.push({
            name: modelName,
            path: modelPath,
            size: size,
            sizeFormatted: formatBytes(size),
            modified: stats.mtime.toISOString(),
            files: modelFiles.length
          });
        } else {
          console.log(`[Models] Modello ${modelName} non ha file validi`);
        }
      } catch (err) {
        // Ignora errori di lettura directory
        console.error(`[Models] Errore lettura ${modelPath}:`, err.message, err.stack);
      }
    }
  }
  
  // Ordina per data di modifica (più recenti prima)
  models.sort((a, b) => new Date(b.modified) - new Date(a.modified));
  
  console.log(`[Models] Totale modelli trovati: ${models.length}`);
  
  return models;
}

// Funzione helper per calcolare dimensione directory
// Usa lstat per non seguire i symlink (evita di contare i file due volte)
function getDirectorySize(dirPath) {
  let totalSize = 0;
  try {
    const files = fs.readdirSync(dirPath);
    for (const file of files) {
      const filePath = path.join(dirPath, file);
      let stats;
      try {
        // Usa lstat per non seguire i symlink
        stats = fs.lstatSync(filePath);
      } catch (err) {
        // Se lstat fallisce, prova stat normale
        try {
          stats = fs.statSync(filePath);
        } catch (err2) {
          continue; // Salta file inaccessibili
        }
      }
      
      // Ignora i symlink (i file reali sono già contati in blobs/)
      if (stats.isSymbolicLink()) {
        continue;
      }
      
      if (stats.isDirectory()) {
        totalSize += getDirectorySize(filePath);
      } else {
        totalSize += stats.size;
      }
    }
  } catch (err) {
    // Ignora errori
  }
  return totalSize;
}

// Funzione helper per formattare bytes
function formatBytes(bytes) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// Scarica modello da HuggingFace
app.post('/api/models/download', async (req, res) => {
  try {
    const { modelId, hfToken } = req.body;
    
    if (!modelId) {
      return res.status(400).json({ error: 'Model ID richiesto' });
    }
    
    // Prepara variabili d'ambiente per il download
    const env = { ...process.env };
    if (hfToken && hfToken.trim() !== '') {
      env.HF_TOKEN = hfToken.trim();
    }
    
    // Usa huggingface-cli per scaricare il modello
    // Prima verifica se è installato, altrimenti usa Python
    const pythonCmd = 'python3.11';
    const downloadScript = `
import os
import sys
from huggingface_hub import snapshot_download

model_id = "${modelId}"
token = os.environ.get('HF_TOKEN', None)

try:
    cache_dir = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
    print(f"Downloading {model_id} to {cache_dir}...")
    snapshot_download(
        repo_id=model_id,
        token=token,
        cache_dir=cache_dir,
        resume_download=True
    )
    print("Download completato con successo!")
except Exception as e:
    print(f"ERRORE: {str(e)}", file=sys.stderr)
    sys.exit(1)
`;
    
    // Scrivi script temporaneo
    const scriptPath = '/tmp/download_model.py';
    fs.writeFileSync(scriptPath, downloadScript);
    
    // Esegui script Python
    const downloadProcess = spawn(pythonCmd, [scriptPath], {
      env: env,
      cwd: '/app'
    });
    
    let stdout = '';
    let stderr = '';
    
    downloadProcess.stdout.on('data', (data) => {
      stdout += data.toString();
      console.log('[Download]', data.toString());
    });
    
    downloadProcess.stderr.on('data', (data) => {
      stderr += data.toString();
      console.error('[Download]', data.toString());
    });
    
    downloadProcess.on('close', (code) => {
      if (code === 0) {
        res.json({ 
          success: true, 
          message: `Modello ${modelId} scaricato con successo`,
          output: stdout
        });
      } else {
        res.status(500).json({ 
          error: `Errore nel download: ${stderr || stdout}`,
          output: stderr || stdout
        });
      }
    });
    
    downloadProcess.on('error', (error) => {
      res.status(500).json({ error: `Errore nell'avvio del download: ${error.message}` });
    });
    
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Elimina modello dalla cache
app.delete('/api/models/delete/:modelName', (req, res) => {
  try {
    const modelName = req.params.modelName;
    // Converti owner/model-name in models--owner--model-name
    const modelDirName = `models--${modelName.replace(/\//g, '--')}`;
    
    // Prova diverse posizioni comuni per la cache HuggingFace
    const possibleCacheDirs = [
      process.env.HF_HOME,
      process.env.HUGGINGFACE_HUB_CACHE,
      '/mnt/huggingface-gds', // Mount della cache dell'utente gds nel container
      '/home/gds/.cache/huggingface',
      '/root/.cache/huggingface',
      path.join(process.env.HOME || '/root', '.cache/huggingface')
    ].filter(Boolean);
    
    let modelPath = null;
    
    // Cerca il modello in tutte le directory possibili
    for (const cacheDir of possibleCacheDirs) {
      // Prova nella directory principale (nuova struttura)
      const mainPath = path.join(cacheDir, modelDirName);
      if (fs.existsSync(mainPath)) {
        modelPath = mainPath;
        break;
      }
      
      // Prova anche in hub/ (vecchia struttura)
      const hubPath = path.join(cacheDir, 'hub', modelDirName);
      if (fs.existsSync(hubPath)) {
        modelPath = hubPath;
        break;
      }
    }
    
    if (!modelPath) {
      return res.status(404).json({ error: 'Modello non trovato nella cache' });
    }
    
    // Elimina directory ricorsivamente
    fs.rmSync(modelPath, { recursive: true, force: true });
    
    res.json({ success: true, message: `Modello ${modelName} eliminato` });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok' });
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`🚀 Backend API server avviato su porta ${PORT}`);
});
