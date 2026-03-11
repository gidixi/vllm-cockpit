import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:3001';
const MAX_HISTORY_POINTS = 100;

function App() {
  const [activeTab, setActiveTab] = useState('config');
  const [status, setStatus] = useState('stopped');
  const [loading, setLoading] = useState(false);
  const [logs, setLogs] = useState([]);
  const [gpus, setGpus] = useState([]);
  const [gpusAvailable, setGpusAvailable] = useState(false);
  const [gpuHistory, setGpuHistory] = useState([]);
  const historyRef = useRef([]);
  const [savedConfigs, setSavedConfigs] = useState([]);
  const [showConfigModal, setShowConfigModal] = useState(false);
  const [configFilename, setConfigFilename] = useState('');
  const [models, setModels] = useState([]);
  const [hfToken, setHfToken] = useState('');
  const [downloadModelId, setDownloadModelId] = useState('');
  const [downloading, setDownloading] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState('');
  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState('');
  const [chatSystemPrompt, setChatSystemPrompt] = useState('Sei un assistente utile e conciso.');
  const [chatTemperature, setChatTemperature] = useState(0.7);
  const [chatMaxTokens, setChatMaxTokens] = useState(512);
  const [chatModel, setChatModel] = useState('');
  const [chatModels, setChatModels] = useState([]);
  const [expandedSections, setExpandedSections] = useState({
    base: true,
    parallelism: false,
    memory: false,
    quantization: false,
    gpu: false,
    kvCache: false,
    tools: false,
  });
  const [config, setConfig] = useState({
    // Base
    model: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    host: '0.0.0.0',
    port: 8000,
    trustRemoteCode: true,
    
    // Parallelismo
    tensorParallelSize: 1,
    pipelineParallelSize: 1,
    dataParallelSize: 1,
    
    // Memoria GPU
    gpuMemoryUtilization: 0.9,
    cpuOffloadGb: 0,
    maxModelLen: 4096,
    kvCacheDtype: 'auto',
    blockSize: 16,
    swapSpace: 4,
    
    // Tipo di dato e quantizzazione
    dtype: 'half',
    quantization: '',
    loadFormat: 'auto',
    quantizationParamPath: '',
    
    // GPU Selection
    device: 'auto',
    distributedExecutorBackend: 'ray',
    workerUseRay: false,
    cudaVisibleDevices: '',
    
    // KV Cache e throughput
    maxNumSeqs: 256,
    maxNumBatchedTokens: 16384,
    enablePrefixCaching: false,
    numGpuBlocksOverride: '',
    
    // Tool calling e multi-modal
    enableAutoToolChoice: false,
    toolCallParser: '',
    servedModelName: '',
    mmEncoderTpMode: '',
    mmProcessorCacheType: '',
    reasoningParser: '',
  });
  const chatListRef = useRef(null);

  useEffect(() => {
    checkStatus();
    fetchGpus();
    fetchSavedConfigs();
    fetchModels();
    const statusInterval = setInterval(checkStatus, 2000);
    const gpuInterval = setInterval(fetchGpus, 2000);
    return () => {
      clearInterval(statusInterval);
      clearInterval(gpuInterval);
    };
  }, []);

  useEffect(() => {
    if (activeTab === 'chat') {
      fetchChatModels();
    }
  }, [activeTab]);

  useEffect(() => {
    if (chatListRef.current) {
      chatListRef.current.scrollTop = chatListRef.current.scrollHeight;
    }
  }, [chatMessages, chatLoading]);

  const fetchSavedConfigs = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/config/list`);
      setSavedConfigs(response.data.configs || []);
    } catch (error) {
      console.error('Errore nel caricamento configurazioni:', error);
    }
  };

  const checkStatus = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/status`);
      setStatus(response.data.status);
      if (response.data.logs) {
        setLogs(response.data.logs.slice(-50));
      }
    } catch (error) {
      setStatus('stopped');
    }
  };

  const fetchGpus = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/gpus`);
      const newGpus = response.data.gpus || [];
      setGpus(newGpus);
      setGpusAvailable(response.data.available || false);

      if (newGpus.length > 0) {
        const timestamp = new Date().toLocaleTimeString();
        const newDataPoint = {
          time: timestamp,
          timestamp: Date.now(),
        };

        newGpus.forEach((gpu, index) => {
          newDataPoint[`gpu${index}_vram`] = gpu.memoryUsed;
          newDataPoint[`gpu${index}_vram_total`] = gpu.memoryTotal;
          newDataPoint[`gpu${index}_temp`] = gpu.temperature;
          newDataPoint[`gpu${index}_util`] = gpu.utilization;
        });

        historyRef.current = [...historyRef.current, newDataPoint].slice(-MAX_HISTORY_POINTS);
        setGpuHistory([...historyRef.current]);
      }
    } catch (error) {
      setGpusAvailable(false);
      setGpus([]);
    }
  };

  const handleStart = async () => {
    setLoading(true);
    try {
      await axios.post(`${API_URL}/api/start`, config);
      setTimeout(checkStatus, 1000);
    } catch (error) {
      alert('Errore nell\'avvio: ' + (error.response?.data?.error || error.message));
    } finally {
      setLoading(false);
    }
  };

  const handleStop = async () => {
    setLoading(true);
    try {
      await axios.post(`${API_URL}/api/stop`);
      setTimeout(checkStatus, 1000);
    } catch (error) {
      alert('Errore nello stop: ' + (error.response?.data?.error || error.message));
    } finally {
      setLoading(false);
    }
  };

  const handleConfigChange = (key, value) => {
    setConfig(prev => {
      // Se è un boolean, toggle
      if (typeof prev[key] === 'boolean') {
        return { ...prev, [key]: !prev[key] };
      }
      // Altrimenti usa il valore fornito
      return { ...prev, [key]: value };
    });
  };

  // Helper per gestire input numerici senza causare re-render
  const handleNumberChange = (key, value, isFloat = false) => {
    const stringValue = value.toString().trim();
    // Se il campo è vuoto o contiene solo segno meno, usa 0 come fallback
    if (stringValue === '' || stringValue === '-') {
      const defaultValue = key === 'gpuMemoryUtilization' ? 0.9 : 
                          key === 'port' ? 8000 : 
                          key === 'maxModelLen' ? 4096 : 0;
      setConfig(prev => ({ ...prev, [key]: defaultValue }));
      return;
    }
    // Converti solo se il valore è valido
    const numValue = isFloat ? parseFloat(stringValue) : parseInt(stringValue, 10);
    if (!isNaN(numValue) && isFinite(numValue)) {
      setConfig(prev => ({ ...prev, [key]: numValue }));
    }
  };

  const toggleSection = (section) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  const handleSaveConfig = async () => {
    if (!configFilename.trim()) {
      alert('Inserisci un nome per la configurazione');
      return;
    }
    
    try {
      const filename = configFilename.endsWith('.json') ? configFilename : `${configFilename}.json`;
      await axios.post(`${API_URL}/api/config/save`, {
        config: config,
        filename: filename
      });
      alert('✅ Configurazione salvata con successo!');
      setShowConfigModal(false);
      setConfigFilename('');
      fetchSavedConfigs();
    } catch (error) {
      alert('Errore nel salvataggio: ' + (error.response?.data?.error || error.message));
    }
  };

  // Configurazione di default completa
  const getDefaultConfig = () => ({
    // Base
    model: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    host: '0.0.0.0',
    port: 8000,
    trustRemoteCode: true,
    
    // Parallelismo
    tensorParallelSize: 1,
    pipelineParallelSize: 1,
    dataParallelSize: 1,
    
    // Memoria GPU
    gpuMemoryUtilization: 0.9,
    cpuOffloadGb: 0,
    maxModelLen: 4096,
    kvCacheDtype: 'auto',
    blockSize: 16,
    swapSpace: 4,
    
    // Tipo di dato e quantizzazione
    dtype: 'half',
    quantization: '',
    loadFormat: 'auto',
    quantizationParamPath: '',
    
    // GPU Selection
    device: 'auto',
    distributedExecutorBackend: 'ray',
    workerUseRay: false,
    cudaVisibleDevices: '',
    
    // KV Cache e throughput
    maxNumSeqs: 256,
    maxNumBatchedTokens: 16384,
    enablePrefixCaching: false,
    numGpuBlocksOverride: '',
    
    // Tool calling e multi-modal
    enableAutoToolChoice: false,
    toolCallParser: '',
    servedModelName: '',
    mmEncoderTpMode: '',
    mmProcessorCacheType: '',
    reasoningParser: '',
  });

  const handleLoadConfig = async (filename) => {
    try {
      const response = await axios.get(`${API_URL}/api/config/load/${filename}`);
      const loadedConfig = response.data.config;
      
      // Merge con i valori di default per assicurarsi che tutti i campi siano presenti
      const defaultConfig = getDefaultConfig();
      const mergedConfig = { ...defaultConfig, ...loadedConfig };
      
      // Forza l'aggiornamento dello stato
      setConfig(mergedConfig);
      
      // Assicurati di essere nella tab configurazione
      setActiveTab('config');
      
      // Scroll alla sezione di configurazione dopo un breve delay
      setTimeout(() => {
        const configCard = document.querySelector('.card-body');
        if (configCard) {
          configCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      }, 100);
      
      alert('✅ Configurazione caricata con successo!');
    } catch (error) {
      alert('Errore nel caricamento: ' + (error.response?.data?.error || error.message));
    }
  };

  const handleDeleteConfig = async (filename) => {
    if (!window.confirm(`Sei sicuro di voler eliminare la configurazione "${filename}"?`)) {
      return;
    }
    
    try {
      await axios.delete(`${API_URL}/api/config/delete/${filename}`);
      alert('✅ Configurazione eliminata!');
      fetchSavedConfigs();
    } catch (error) {
      alert('Errore nell\'eliminazione: ' + (error.response?.data?.error || error.message));
    }
  };

  const fetchModels = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/models/list`);
      setModels(response.data.models || []);
    } catch (error) {
      console.error('Errore nel caricamento modelli:', error);
      setModels([]);
    }
  };

  const handleDownloadModel = async () => {
    if (!downloadModelId.trim()) {
      alert('Inserisci un Model ID (es. TinyLlama/TinyLlama-1.1B-Chat-v1.0)');
      return;
    }
    
    setDownloading(true);
    setDownloadProgress('Avvio download...');
    
    try {
      const response = await axios.post(`${API_URL}/api/models/download`, {
        modelId: downloadModelId.trim(),
        hfToken: hfToken.trim() || undefined
      });
      
      alert('✅ Download avviato! Controlla i log per il progresso.');
      setDownloadModelId('');
      setDownloadProgress('');
      
      // Ricarica lista modelli dopo un po'
      setTimeout(() => {
        fetchModels();
      }, 5000);
    } catch (error) {
      alert('Errore nel download: ' + (error.response?.data?.error || error.message));
      setDownloadProgress('');
    } finally {
      setDownloading(false);
    }
  };

  const handleDeleteModel = async (modelName) => {
    if (!window.confirm(`Sei sicuro di voler eliminare il modello "${modelName}"?\n\nQuesta operazione eliminerà tutti i file del modello dalla cache.`)) {
      return;
    }
    
    try {
      await axios.delete(`${API_URL}/api/models/delete/${encodeURIComponent(modelName)}`);
      alert('✅ Modello eliminato!');
      fetchModels();
    } catch (error) {
      alert('Errore nell\'eliminazione: ' + (error.response?.data?.error || error.message));
    }
  };

  const handleUseModel = (modelName) => {
    setConfig(prev => ({ ...prev, model: modelName }));
    setActiveTab('config');
    
    // Scroll alla sezione di configurazione dopo un breve delay
    setTimeout(() => {
      const configCard = document.querySelector('.card-body');
      if (configCard) {
        configCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }, 100);
  };

  const handleDownloadConfig = () => {
    const dataStr = JSON.stringify(config, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `vllm-config-${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const fetchChatModels = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/vllm/models`);
      const availableModels = response?.data?.data || [];
      setChatModels(availableModels);
      if (!chatModel && availableModels.length > 0) {
        setChatModel(availableModels[0].id);
      }
      setChatError('');
    } catch (error) {
      setChatModels([]);
      setChatError(error.response?.data?.error || 'Impossibile leggere i modelli da vLLM. Hai avviato il server?');
    }
  };

  const handleSendMessage = async () => {
    const prompt = chatInput.trim();
    if (!prompt || chatLoading) return;

    const nextMessages = [...chatMessages, { role: 'user', content: prompt }];
    setChatMessages(nextMessages);
    setChatInput('');
    setChatLoading(true);
    setChatError('');

    try {
      const response = await axios.post(`${API_URL}/api/chat`, {
        messages: nextMessages,
        systemPrompt: chatSystemPrompt,
        model: chatModel,
        temperature: chatTemperature,
        max_tokens: chatMaxTokens,
      });

      const assistantText = response?.data?.choices?.[0]?.message?.content;
      const replyStats = response?.data?._proxyStats || null;
      if (!assistantText) {
        throw new Error('Risposta vuota dal modello');
      }

      setChatMessages((prev) => [...prev, { role: 'assistant', content: assistantText, stats: replyStats }]);
    } catch (error) {
      setChatError(error.response?.data?.error || error.response?.data?.details?.error?.message || error.message);
    } finally {
      setChatLoading(false);
    }
  };

  const getChatTotals = () => {
    return chatMessages.reduce((acc, msg) => {
      if (msg.role !== 'assistant' || !msg.stats || !msg.stats.usage) return acc;
      acc.prompt += msg.stats.usage.promptTokens || 0;
      acc.completion += msg.stats.usage.completionTokens || 0;
      acc.total += msg.stats.usage.totalTokens || 0;
      return acc;
    }, { prompt: 0, completion: 0, total: 0 });
  };

  const MarkdownMessage = ({ content }) => (
    <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
  );

  const handleUploadConfig = (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const loadedConfig = JSON.parse(e.target.result);
        
        // Merge con i valori di default per assicurarsi che tutti i campi siano presenti
        const defaultConfig = getDefaultConfig();
        const mergedConfig = { ...defaultConfig, ...loadedConfig };
        
        setConfig(mergedConfig);
        
        // Assicurati di essere nella tab configurazione
        setActiveTab('config');
        
        // Scroll alla sezione di configurazione dopo un breve delay
        setTimeout(() => {
          const configCard = document.querySelector('.card-body');
          if (configCard) {
            configCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
          }
        }, 100);
        
        alert('✅ Configurazione caricata dal file!');
      } catch (error) {
        alert('Errore: File JSON non valido');
      }
    };
    reader.readAsText(file);
  };

  const getStatusText = () => {
    switch (status) {
      case 'running':
        return '🟢 Running';
      case 'starting':
        return '🟡 Starting';
      case 'stopping':
        return '🟠 Stopping';
      default:
        return '🔴 Stopped';
    }
  };

  const getMemoryPercentage = (used, total) => {
    if (!total || total === 0) return 0;
    return Math.round((used / total) * 100);
  };

  const getProgressBarClass = (percentage) => {
    if (percentage >= 90) return 'danger';
    if (percentage >= 70) return 'warning';
    return '';
  };

  const getChartData = () => {
    return gpuHistory.map(point => {
      const chartPoint = { time: point.time };
      gpus.forEach((gpu, index) => {
        chartPoint[`GPU ${index} VRAM (MB)`] = point[`gpu${index}_vram`] || 0;
        chartPoint[`GPU ${index} Temp (°C)`] = point[`gpu${index}_temp`] || 0;
      });
      return chartPoint;
    });
  };

  const getChartColors = () => {
    const colors = [
      { vram: '#3b82f6', temp: '#ef4444' },
      { vram: '#10b981', temp: '#f59e0b' },
      { vram: '#8b5cf6', temp: '#ec4899' },
      { vram: '#06b6d4', temp: '#f97316' },
    ];
    return colors;
  };

  const ConfigSection = ({ title, sectionKey, children }) => (
    <div className="config-section">
      <div className="config-section-header" onClick={() => toggleSection(sectionKey)}>
        <h3>{title}</h3>
        <span className="section-toggle">{expandedSections[sectionKey] ? '▼' : '▶'}</span>
      </div>
      {expandedSections[sectionKey] && (
        <div className="config-section-content">
          {children}
        </div>
      )}
    </div>
  );

  return (
    <div className="wrapper">
      {/* Navbar */}
      <nav className="main-header navbar navbar-expand navbar-white navbar-light">
        <ul className="navbar-nav">
          <li className="nav-item">
            <a className="nav-link" data-widget="pushmenu" href="#" role="button">
              <i className="fas fa-bars"></i>
            </a>
          </li>
        </ul>
        <ul className="navbar-nav ml-auto">
          <li className="nav-item">
            <span className={`nav-link status-indicator ${status}`}>
              <i className={`fas fa-circle ${status === 'running' ? 'text-success' : status === 'starting' ? 'text-warning' : 'text-danger'}`}></i>
              <span className="ml-2">{getStatusText()}</span>
            </span>
          </li>
        </ul>
      </nav>

      {/* Sidebar */}
      <aside className="main-sidebar sidebar-dark-primary elevation-4">
        <a href="#" className="brand-link">
          <i className="fas fa-robot brand-image" style={{ fontSize: '2rem', marginRight: '0.5rem' }}></i>
          <span className="brand-text font-weight-light">vLLM Control</span>
        </a>
        <div className="sidebar">
          <nav className="mt-2">
            <ul className="nav nav-pills nav-sidebar flex-column" data-widget="treeview" role="menu">
              <li className="nav-item">
                <a 
                  href="#" 
                  className={`nav-link ${activeTab === 'config' ? 'active' : ''}`}
                  onClick={(e) => { e.preventDefault(); setActiveTab('config'); }}
                >
                  <i className="nav-icon fas fa-cog"></i>
                  <p>Configurazione</p>
                </a>
              </li>
              <li className="nav-item">
                <a 
                  href="#" 
                  className={`nav-link ${activeTab === 'models' ? 'active' : ''}`}
                  onClick={(e) => { e.preventDefault(); setActiveTab('models'); }}
                >
                  <i className="nav-icon fas fa-database"></i>
                  <p>Modelli</p>
                </a>
              </li>
              <li className="nav-item">
                <a 
                  href="#" 
                  className={`nav-link ${activeTab === 'monitor' ? 'active' : ''}`}
                  onClick={(e) => { e.preventDefault(); setActiveTab('monitor'); }}
                >
                  <i className="nav-icon fas fa-chart-line"></i>
                  <p>Monitoraggio GPU</p>
                </a>
              </li>
              <li className="nav-item">
                <a 
                  href="#" 
                  className={`nav-link ${activeTab === 'logs' ? 'active' : ''}`}
                  onClick={(e) => { e.preventDefault(); setActiveTab('logs'); }}
                >
                  <i className="nav-icon fas fa-file-alt"></i>
                  <p>Logs</p>
                </a>
              </li>
              <li className="nav-item">
                <a
                  href="#"
                  className={`nav-link ${activeTab === 'chat' ? 'active' : ''}`}
                  onClick={(e) => { e.preventDefault(); setActiveTab('chat'); }}
                >
                  <i className="nav-icon fas fa-comments"></i>
                  <p>Chat Test</p>
                </a>
              </li>
            </ul>
          </nav>
        </div>
      </aside>

      {/* Content Wrapper */}
      <div className="content-wrapper">
        {/* Content Header */}
        <div className="content-header">
          <div className="container-fluid">
            <div className="row mb-2">
              <div className="col-sm-6">
                <h1 className="m-0">
                  {activeTab === 'config' && '⚙️ Configurazione vLLM'}
                  {activeTab === 'models' && '🤖 Gestione Modelli'}
                  {activeTab === 'monitor' && '📊 Monitoraggio GPU'}
                  {activeTab === 'logs' && '📋 Logs vLLM'}
                  {activeTab === 'chat' && '💬 Chat Test Modello'}
                </h1>
              </div>
              <div className="col-sm-6">
                <ol className="breadcrumb float-sm-right">
                  <li className="breadcrumb-item"><a href="#">Home</a></li>
                  <li className="breadcrumb-item active">
                    {activeTab === 'config' && 'Configurazione'}
                    {activeTab === 'models' && 'Modelli'}
                    {activeTab === 'monitor' && 'Monitoraggio'}
                    {activeTab === 'logs' && 'Logs'}
                    {activeTab === 'chat' && 'Chat Test'}
                  </li>
                </ol>
              </div>
            </div>
          </div>
        </div>

        {/* Main content */}
        <section className="content">
          <div className="container-fluid">
            <React.Fragment>
            {activeTab === 'config' && (
              <div className="row">
                <div className="col-12">
                  <div className="card">
                    <div className="card-header">
                      <h3 className="card-title">
                        <i className="fas fa-cog mr-2"></i>
                        Configurazione vLLM
                      </h3>
                      <div className="card-tools">
                        <button
                          onClick={() => setShowConfigModal(true)}
                          className="btn btn-sm btn-primary mr-2"
                          title="Salva configurazione"
                        >
                          <i className="fas fa-save"></i> Salva
                        </button>
                        <label className="btn btn-sm btn-info mr-2" title="Carica da file locale">
                          <i className="fas fa-folder-open"></i> Carica File
                          <input
                            type="file"
                            accept=".json"
                            onChange={handleUploadConfig}
                            style={{ display: 'none' }}
                          />
                        </label>
                        <button
                          onClick={handleDownloadConfig}
                          className="btn btn-sm btn-success"
                          title="Scarica configurazione"
                        >
                          <i className="fas fa-download"></i> Scarica
                        </button>
                      </div>
                    </div>
                    <div className="card-body">
                      {/* Lista configurazioni salvate */}
                      {savedConfigs.length > 0 && (
                        <div className="card card-info card-outline mb-3">
                          <div className="card-header">
                            <h3 className="card-title">
                              <i className="fas fa-book mr-2"></i>
                              Configurazioni Salvate
                            </h3>
                          </div>
                          <div className="card-body p-0">
                            <table className="table table-striped">
                              <thead>
                                <tr>
                                  <th>Nome</th>
                                  <th>Data Modifica</th>
                                  <th style={{ width: '150px' }}>Azioni</th>
                                </tr>
                              </thead>
                              <tbody>
                                {savedConfigs.map((savedConfig, index) => (
                                  <tr key={index}>
                                    <td>{savedConfig.filename}</td>
                                    <td>{new Date(savedConfig.modified).toLocaleString()}</td>
                                    <td>
                                      <button
                                        onClick={() => handleLoadConfig(savedConfig.filename)}
                                        className="btn btn-sm btn-info mr-1"
                                      >
                                        <i className="fas fa-folder-open"></i> Carica
                                      </button>
                                      <button
                                        onClick={() => handleDeleteConfig(savedConfig.filename)}
                                        className="btn btn-sm btn-danger"
                                      >
                                        <i className="fas fa-trash"></i>
                                      </button>
                                    </td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>
                      )}
              
              <ConfigSection title="📋 Base" sectionKey="base">
                <div className="form-group">
                  <label>Modello:</label>
                  <input
                    type="text"
                    value={config.model}
                    onChange={(e) => handleConfigChange('model', e.target.value)}
                    placeholder="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                  />
                </div>
                <div className="form-row">
                  <div className="form-group">
                    <label>Host:</label>
                    <input
                      type="text"
                      value={config.host}
                      onChange={(e) => handleConfigChange('host', e.target.value)}
                    />
                  </div>
                  <div className="form-group">
                    <label>Porta:</label>
                    <input
                      type="number"
                      value={config.port}
                      onChange={(e) => handleNumberChange('port', e.target.value)}
                    />
                  </div>
                </div>
                <div className="form-group checkbox">
                  <label>
                    <input
                      type="checkbox"
                      checked={config.trustRemoteCode}
                      onChange={() => handleConfigChange('trustRemoteCode')}
                    />
                    Trust Remote Code
                  </label>
                </div>
              </ConfigSection>

              <ConfigSection title="🔄 Parallelismo e Distribuzione" sectionKey="parallelism">
                <div className="form-row">
                  <div className="form-group">
                    <label>Tensor Parallel Size:</label>
                    <input
                      type="number"
                      min="1"
                      value={config.tensorParallelSize}
                      onChange={(e) => handleNumberChange('tensorParallelSize', e.target.value)}
                    />
                    <small>Numero di GPU per tensor parallelism (intra-nodo)</small>
                    {config.tensorParallelSize > 1 && gpus.length > 0 && config.tensorParallelSize > gpus.length && (
                      <div className="alert alert-warning mt-2" style={{ padding: '0.5rem', fontSize: '0.875rem' }}>
                        <i className="fas fa-exclamation-triangle mr-1"></i>
                        <strong>Attenzione:</strong> Tensor Parallel Size ({config.tensorParallelSize}) è maggiore del numero di GPU disponibili ({gpus.length}). 
                        Questo causerà un errore all'avvio. Riduci il valore a {gpus.length} o meno.
                      </div>
                    )}
                  </div>
                  <div className="form-group">
                    <label>Pipeline Parallel Size:</label>
                    <input
                      type="number"
                      min="1"
                      value={config.pipelineParallelSize}
                      onChange={(e) => handleNumberChange('pipelineParallelSize', e.target.value)}
                    />
                    <small>Numero di nodi per pipeline parallelism (inter-nodo)</small>
                  </div>
                </div>
                <div className="form-group">
                  <label>Data Parallel Size:</label>
                  <input
                    type="number"
                    min="1"
                    value={config.dataParallelSize}
                    onChange={(e) => handleNumberChange('dataParallelSize', e.target.value)}
                  />
                  <small>Replica il modello su più istanze per aumentare throughput</small>
                </div>
              </ConfigSection>

              <ConfigSection title="💾 Gestione Memoria GPU" sectionKey="memory">
                <div className="form-row">
                  <div className="form-group">
                    <label>GPU Memory Utilization:</label>
                    <input
                      type="number"
                      step="0.01"
                      min="0"
                      max="1"
                      value={config.gpuMemoryUtilization}
                      onChange={(e) => handleNumberChange('gpuMemoryUtilization', e.target.value, true)}
                    />
                    <small>Frazione di VRAM usata (0.0-1.0)</small>
                  </div>
                  <div className="form-group">
                    <label>CPU Offload (GB):</label>
                    <input
                      type="number"
                      min="0"
                      value={config.cpuOffloadGb}
                      onChange={(e) => handleNumberChange('cpuOffloadGb', e.target.value)}
                    />
                    <small>GiB da offloadare su RAM CPU</small>
                  </div>
                </div>
                <div className="form-row">
                  <div className="form-group">
                    <label>Max Model Length:</label>
                    <input
                      type="number"
                      value={config.maxModelLen}
                      onChange={(e) => handleNumberChange('maxModelLen', e.target.value)}
                    />
                    <small>Lunghezza massima del contesto</small>
                  </div>
                  <div className="form-group">
                    <label>KV Cache Dtype:</label>
                    <select
                      value={config.kvCacheDtype}
                      onChange={(e) => handleConfigChange('kvCacheDtype', e.target.value)}
                    >
                      <option value="auto">auto</option>
                      <option value="fp8">fp8</option>
                      <option value="fp16">fp16</option>
                      <option value="fp32">fp32</option>
                    </select>
                  </div>
                </div>
                <div className="form-row">
                  <div className="form-group">
                    <label>Block Size:</label>
                    <input
                      type="number"
                      min="1"
                      value={config.blockSize}
                      onChange={(e) => handleNumberChange('blockSize', e.target.value)}
                    />
                    <small>Dimensione blocchi per paging KV cache</small>
                  </div>
                  <div className="form-group">
                    <label>Swap Space (GB):</label>
                    <input
                      type="number"
                      min="0"
                      value={config.swapSpace}
                      onChange={(e) => handleNumberChange('swapSpace', e.target.value)}
                    />
                    <small>GiB di swap su CPU per blocchi evicted</small>
                  </div>
                </div>
              </ConfigSection>

              <ConfigSection title="🔢 Tipo di Dato e Quantizzazione" sectionKey="quantization">
                <div className="form-row">
                  <div className="form-group">
                    <label>Data Type (Dtype):</label>
                    <select
                      value={config.dtype}
                      onChange={(e) => handleConfigChange('dtype', e.target.value)}
                    >
                      <option value="auto">auto</option>
                      <option value="half">half (FP16)</option>
                      <option value="float16">float16</option>
                      <option value="bfloat16">bfloat16</option>
                      <option value="float32">float32</option>
                    </select>
                  </div>
                  <div className="form-group">
                    <label>Quantization:</label>
                    <select
                      value={config.quantization}
                      onChange={(e) => handleConfigChange('quantization', e.target.value)}
                    >
                      <option value="">None</option>
                      <option value="awq">AWQ</option>
                      <option value="gptq">GPTQ</option>
                      <option value="squeezellm">SqueezeLLM</option>
                      <option value="fp8">FP8</option>
                    </select>
                  </div>
                </div>
                <div className="form-row">
                  <div className="form-group">
                    <label>Load Format:</label>
                    <select
                      value={config.loadFormat}
                      onChange={(e) => handleConfigChange('loadFormat', e.target.value)}
                    >
                      <option value="auto">auto</option>
                      <option value="pt">pt</option>
                      <option value="safetensors">safetensors</option>
                      <option value="npcache">npcache</option>
                      <option value="dummy">dummy</option>
                    </select>
                  </div>
                  <div className="form-group">
                    <label>Quantization Param Path:</label>
                    <input
                      type="text"
                      value={config.quantizationParamPath}
                      onChange={(e) => handleConfigChange('quantizationParamPath', e.target.value)}
                      placeholder="Path ai parametri di quantizzazione"
                    />
                  </div>
                </div>
              </ConfigSection>

              <ConfigSection title="🎯 Selezione e Visibilità GPU" sectionKey="gpu">
                <div className="form-row">
                  <div className="form-group">
                    <label>Device:</label>
                    <select
                      value={config.device}
                      onChange={(e) => handleConfigChange('device', e.target.value)}
                    >
                      <option value="auto">auto</option>
                      <option value="cuda">cuda</option>
                      <option value="neuron">neuron</option>
                      <option value="cpu">cpu</option>
                    </select>
                  </div>
                  <div className="form-group">
                    <label>Distributed Executor Backend:</label>
                    <select
                      value={config.distributedExecutorBackend}
                      onChange={(e) => handleConfigChange('distributedExecutorBackend', e.target.value)}
                    >
                      <option value="ray">ray</option>
                      <option value="mp">mp (multiprocessing)</option>
                    </select>
                  </div>
                </div>
                <div className="form-row">
                  <div className="form-group">
                    <label>CUDA Visible Devices:</label>
                    <input
                      type="text"
                      value={config.cudaVisibleDevices}
                      onChange={(e) => handleConfigChange('cudaVisibleDevices', e.target.value)}
                      placeholder="0,1,2,3"
                    />
                    <small>Es: 0,1,2,3 per selezionare GPU specifiche</small>
                  </div>
                  <div className="form-group checkbox">
                    <label>
                      <input
                        type="checkbox"
                        checked={config.workerUseRay}
                        onChange={() => handleConfigChange('workerUseRay')}
                      />
                      Worker Use Ray
                    </label>
                    <small>Forza l'uso di Ray come worker backend</small>
                  </div>
                </div>
              </ConfigSection>

              <ConfigSection title="⚡ KV Cache e Throughput" sectionKey="kvCache">
                <div className="form-row">
                  <div className="form-group">
                    <label>Max Num Seqs:</label>
                    <input
                      type="number"
                      min="1"
                      value={config.maxNumSeqs}
                      onChange={(e) => handleNumberChange('maxNumSeqs', e.target.value)}
                    />
                    <small>Numero massimo di sequenze in batch simultaneo</small>
                  </div>
                  <div className="form-group">
                    <label>Max Batched Tokens:</label>
                    <input
                      type="number"
                      min="1"
                      value={config.maxNumBatchedTokens}
                      onChange={(e) => handleNumberChange('maxNumBatchedTokens', e.target.value)}
                    />
                    <small>Token massimi per batch (influisce su throughput e VRAM)</small>
                  </div>
                </div>
                <div className="form-row">
                  <div className="form-group">
                    <label>Num GPU Blocks Override:</label>
                    <input
                      type="text"
                      value={config.numGpuBlocksOverride}
                      onChange={(e) => handleConfigChange('numGpuBlocksOverride', e.target.value)}
                      placeholder="Override manuale blocchi GPU"
                    />
                    <small>Lascia vuoto per auto</small>
                  </div>
                  <div className="form-group checkbox">
                    <label>
                      <input
                        type="checkbox"
                        checked={config.enablePrefixCaching}
                        onChange={() => handleConfigChange('enablePrefixCaching')}
                      />
                      Enable Prefix Caching
                    </label>
                    <small>Cache dei prefix condivisi (risparmia VRAM e latenza)</small>
                  </div>
                </div>
              </ConfigSection>

              <ConfigSection title="🔧 Tool Calling e Multi-Modal (Opzionale)" sectionKey="tools">
                <div className="form-row">
                  <div className="form-group checkbox">
                    <label>
                      <input
                        type="checkbox"
                        checked={config.enableAutoToolChoice}
                        onChange={() => handleConfigChange('enableAutoToolChoice')}
                      />
                      Enable Auto Tool Choice
                    </label>
                    <small>Abilita la scelta automatica degli strumenti per i modelli che supportano tool calling</small>
                  </div>
                  <div className="form-group">
                    <label>Tool Call Parser:</label>
                    <select
                      value={config.toolCallParser}
                      onChange={(e) => handleConfigChange('toolCallParser', e.target.value)}
                    >
                      <option value="">Auto (default)</option>
                      <option value="hermes">hermes</option>
                      <option value="qwen3_coder">qwen3_coder</option>
                      <option value="qwen3">qwen3</option>
                    </select>
                    <small>Parser per interpretare le chiamate agli strumenti</small>
                  </div>
                </div>
                <div className="form-row">
                  <div className="form-group">
                    <label>Served Model Name:</label>
                    <input
                      type="text"
                      value={config.servedModelName}
                      onChange={(e) => handleConfigChange('servedModelName', e.target.value)}
                      placeholder="Nome del modello esposto via API"
                    />
                    <small>Nome del modello come appare nell'API (lascia vuoto per usare il nome del modello)</small>
                  </div>
                  <div className="form-group">
                    <label>Reasoning Parser:</label>
                    <select
                      value={config.reasoningParser}
                      onChange={(e) => handleConfigChange('reasoningParser', e.target.value)}
                    >
                      <option value="">Nessuno</option>
                      <option value="qwen3">qwen3</option>
                    </select>
                    <small>Parser per il reasoning (es. qwen3 per modelli Qwen3)</small>
                  </div>
                </div>
                <div className="form-row">
                  <div className="form-group">
                    <label>MM Encoder TP Mode:</label>
                    <select
                      value={config.mmEncoderTpMode}
                      onChange={(e) => handleConfigChange('mmEncoderTpMode', e.target.value)}
                    >
                      <option value="">Nessuno</option>
                      <option value="data">data</option>
                      <option value="tensor">tensor</option>
                    </select>
                    <small>Modalità tensor parallelism per encoder multi-modale</small>
                  </div>
                  <div className="form-group">
                    <label>MM Processor Cache Type:</label>
                    <select
                      value={config.mmProcessorCacheType}
                      onChange={(e) => handleConfigChange('mmProcessorCacheType', e.target.value)}
                    >
                      <option value="">Nessuno</option>
                      <option value="shm">shm (shared memory)</option>
                      <option value="cpu">cpu</option>
                    </select>
                    <small>Tipo di cache per il processore multi-modale</small>
                  </div>
                </div>
              </ConfigSection>

              <div className="button-group">
                <button
                  onClick={handleStart}
                  disabled={loading || status === 'running' || status === 'starting'}
                  className="btn btn-start"
                >
                  ▶️ Avvia vLLM
                </button>
                <button
                  onClick={handleStop}
                  disabled={loading || status === 'stopped'}
                  className="btn btn-stop"
                >
                  ⏹️ Ferma vLLM
                </button>
              </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

          {activeTab === 'models' && (
            <div className="row">
              <div className="col-12">
                <div className="card">
                  <div className="card-header">
                    <h3 className="card-title">
                      <i className="fas fa-download mr-2"></i>
                      Scarica Modello da HuggingFace
                    </h3>
                  </div>
                  <div className="card-body">
                    <div className="form-group">
                      <label>HuggingFace Token (opzionale):</label>
                      <input
                        type="password"
                        className="form-control"
                        value={hfToken}
                        onChange={(e) => setHfToken(e.target.value)}
                        placeholder="hf_xxxxxxxxxxxxx"
                      />
                      <small className="form-text text-muted">Token necessario per modelli privati o gated. Ottienilo da <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener noreferrer">qui</a></small>
                    </div>
                    <div className="form-group">
                      <label>Model ID:</label>
                      <input
                        type="text"
                        className="form-control"
                        value={downloadModelId}
                        onChange={(e) => setDownloadModelId(e.target.value)}
                        placeholder="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                        disabled={downloading}
                      />
                      <small className="form-text text-muted">Formato: owner/model-name (es. meta-llama/Llama-3.1-70B-Instruct)</small>
                    </div>
                    <button
                      onClick={handleDownloadModel}
                      disabled={downloading || !downloadModelId.trim()}
                      className="btn btn-primary"
                    >
                      <i className="fas fa-download mr-2"></i>
                      {downloading ? 'Download in corso...' : 'Scarica Modello'}
                    </button>
                    {downloadProgress && (
                      <div className="alert alert-info mt-3">{downloadProgress}</div>
                    )}
                  </div>
                </div>
              </div>
              
              <div className="col-12 mt-3">
                <div className="card">
                  <div className="card-header">
                    <h3 className="card-title">
                      <i className="fas fa-database mr-2"></i>
                      Modelli Disponibili ({models.length})
                    </h3>
                    <div className="card-tools">
                      <button onClick={fetchModels} className="btn btn-sm btn-primary">
                        <i className="fas fa-sync-alt"></i> Aggiorna
                      </button>
                    </div>
                  </div>
                  <div className="card-body">
                    {models.length === 0 ? (
                      <div className="alert alert-info">
                        <i className="fas fa-info-circle mr-2"></i>
                        Nessun modello trovato nella cache HuggingFace. Scarica un modello usando il form sopra.
                      </div>
                    ) : (
                      <div className="row">
                        {models.map((model, index) => (
                          <div key={index} className="col-md-6 col-lg-4 mb-3">
                            <div className="card card-outline card-primary">
                              <div className="card-header">
                                <h3 className="card-title">{model.name}</h3>
                                <div className="card-tools">
                                  <button
                                    onClick={() => handleUseModel(model.name)}
                                    className="btn btn-sm btn-success mr-1"
                                    title="Usa questo modello"
                                  >
                                    <i className="fas fa-check"></i>
                                  </button>
                                  <button
                                    onClick={() => handleDeleteModel(model.name)}
                                    className="btn btn-sm btn-danger"
                                    title="Elimina modello"
                                  >
                                    <i className="fas fa-trash"></i>
                                  </button>
                                </div>
                              </div>
                              <div className="card-body">
                                <dl className="row mb-0">
                                  <dt className="col-sm-5">Dimensione:</dt>
                                  <dd className="col-sm-7">{model.sizeFormatted}</dd>
                                  <dt className="col-sm-5">File:</dt>
                                  <dd className="col-sm-7">{model.files}</dd>
                                  <dt className="col-sm-5">Modificato:</dt>
                                  <dd className="col-sm-7">{new Date(model.modified).toLocaleString('it-IT')}</dd>
                                </dl>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'monitor' && (
            <div className="row">
              <div className="col-12">
                <div className="card">
                  <div className="card-header">
                    <h3 className="card-title">
                      <i className="fas fa-chart-line mr-2"></i>
                      Monitoraggio GPU
                    </h3>
                  </div>
                  <div className="card-body">
                    {gpusAvailable && gpus.length > 0 ? (
                      <>
                        <div className="chart-container">
                          <h4 className="mb-3">VRAM e Temperatura nel tempo</h4>
                          <ResponsiveContainer width="100%" height={400}>
                      <LineChart data={getChartData()}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis 
                          dataKey="time" 
                          tick={{ fontSize: 12 }}
                          interval="preserveStartEnd"
                        />
                        <YAxis yAxisId="left" label={{ value: 'VRAM (MB)', angle: -90, position: 'insideLeft' }} />
                        <YAxis yAxisId="right" orientation="right" label={{ value: 'Temp (°C)', angle: 90, position: 'insideRight' }} />
                        <Tooltip />
                        <Legend />
                        {gpus.map((gpu, index) => {
                          const colors = getChartColors()[index % getChartColors().length];
                          return (
                            <React.Fragment key={index}>
                              <Line 
                                yAxisId="left"
                                type="monotone" 
                                dataKey={`GPU ${index} VRAM (MB)`} 
                                stroke={colors.vram} 
                                strokeWidth={2}
                                dot={false}
                                name={`GPU ${index} VRAM`}
                              />
                              <Line 
                                yAxisId="right"
                                type="monotone" 
                                dataKey={`GPU ${index} Temp (°C)`} 
                                stroke={colors.temp} 
                                strokeWidth={2}
                                dot={false}
                                strokeDasharray="5 5"
                                name={`GPU ${index} Temp`}
                              />
                            </React.Fragment>
                          );
                        })}
                      </LineChart>
                    </ResponsiveContainer>
                        </div>

                        <div className="row mt-4">
                          {gpus.map((gpu) => {
                            const memoryPercent = getMemoryPercentage(gpu.memoryUsed, gpu.memoryTotal);
                            return (
                              <div key={gpu.index} className="col-md-6 col-lg-4 mb-3">
                                <div className="card card-outline card-info">
                                  <div className="card-header">
                                    <h3 className="card-title">
                                      <i className="fas fa-microchip mr-2"></i>
                                      {gpu.name} (GPU {gpu.index})
                                    </h3>
                                  </div>
                                  <div className="card-body">
                                    <dl className="row mb-3">
                                      <dt className="col-sm-6">Utilizzo:</dt>
                                      <dd className="col-sm-6">{gpu.utilization}%</dd>
                                      <dt className="col-sm-6">Temperatura:</dt>
                                      <dd className="col-sm-6">{gpu.temperature}°C</dd>
                                      <dt className="col-sm-6">Memoria Usata:</dt>
                                      <dd className="col-sm-6">{gpu.memoryUsed} MB</dd>
                                      <dt className="col-sm-6">Memoria Totale:</dt>
                                      <dd className="col-sm-6">{gpu.memoryTotal} MB</dd>
                                    </dl>
                                    <div className="progress mb-2" style={{ height: '20px' }}>
                                      <div 
                                        className={`progress-bar ${memoryPercent >= 90 ? 'bg-danger' : memoryPercent >= 70 ? 'bg-warning' : 'bg-success'}`}
                                        role="progressbar"
                                        style={{ width: `${memoryPercent}%` }}
                                        aria-valuenow={memoryPercent}
                                        aria-valuemin="0"
                                        aria-valuemax="100"
                                      >
                                        {memoryPercent}%
                                      </div>
                                    </div>
                                    <small className="text-muted">Memoria utilizzata: {memoryPercent}%</small>
                                  </div>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </>
                    ) : (
                      <div className="alert alert-warning">
                        <i className="fas fa-exclamation-triangle mr-2"></i>
                        <strong>Nessuna GPU disponibile</strong>
                        <p className="mb-0 mt-2">
                          Assicurati che nvidia-smi sia installato e che le GPU siano accessibili.
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'logs' && (
            <div className="row">
              <div className="col-12">
                <div className="card">
                  <div className="card-header">
                    <h3 className="card-title">
                      <i className="fas fa-file-alt mr-2"></i>
                      Logs vLLM
                    </h3>
                  </div>
                  <div className="card-body p-0">
                    <div className="logs-container">
                      {logs.length === 0 ? (
                        <div className="alert alert-info m-3">
                          <i className="fas fa-info-circle mr-2"></i>
                          Nessun log disponibile
                        </div>
                      ) : (
                        logs.map((log, index) => (
                          <div key={index} className="log-line">
                            {log}
                          </div>
                        ))
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'chat' && (
            <div className="row">
              <div className="col-12">
                <div className="card">
                  <div className="card-header">
                    <h3 className="card-title">
                      <i className="fas fa-comments mr-2"></i>
                      Chat di Test vLLM
                    </h3>
                    <div className="card-tools">
                      <button onClick={fetchChatModels} className="btn btn-sm btn-primary">
                        <i className="fas fa-sync-alt"></i> Aggiorna Modelli
                      </button>
                    </div>
                  </div>
                  <div className="card-body">
                    <div className="form-row">
                      <div className="form-group">
                        <label>Modello:</label>
                        <select value={chatModel} onChange={(e) => setChatModel(e.target.value)}>
                          {chatModels.map((m) => (
                            <option key={m.id} value={m.id}>{m.id}</option>
                          ))}
                        </select>
                        <small>Viene letta la lista direttamente da `vLLM /v1/models`.</small>
                      </div>
                      <div className="form-group">
                        <label>Temperature:</label>
                        <input
                          type="number"
                          min="0"
                          max="2"
                          step="0.1"
                          value={chatTemperature}
                          onChange={(e) => setChatTemperature(parseFloat(e.target.value) || 0)}
                        />
                      </div>
                      <div className="form-group">
                        <label>Max Tokens:</label>
                        <input
                          type="number"
                          min="1"
                          value={chatMaxTokens}
                          onChange={(e) => setChatMaxTokens(parseInt(e.target.value, 10) || 1)}
                        />
                      </div>
                    </div>

                    <div className="form-group">
                      <label>System Prompt (opzionale):</label>
                      <input
                        type="text"
                        value={chatSystemPrompt}
                        onChange={(e) => setChatSystemPrompt(e.target.value)}
                        placeholder="Istruzioni di comportamento per l'assistente"
                      />
                    </div>

                    {chatError && (
                      <div className="alert alert-danger">
                        <i className="fas fa-exclamation-circle mr-2"></i>
                        {chatError}
                      </div>
                    )}

                    <div className="chat-stats-summary">
                      {(() => {
                        const totals = getChatTotals();
                        return (
                          <>
                            <span><strong>Token prompt:</strong> {totals.prompt}</span>
                            <span><strong>Token output:</strong> {totals.completion}</span>
                            <span><strong>Token totali:</strong> {totals.total}</span>
                          </>
                        );
                      })()}
                    </div>

                    <div className="chat-box" ref={chatListRef}>
                      {chatMessages.length === 0 ? (
                        <div className="chat-empty">
                          Scrivi un messaggio per testare il modello.
                        </div>
                      ) : (
                        chatMessages.map((message, index) => (
                          <div key={index} className={`chat-message ${message.role}`}>
                            <div className="chat-role">
                              {message.role === 'user' ? 'Tu' : 'Assistente'}
                            </div>
                            <div className="chat-content markdown-body">
                              <MarkdownMessage content={message.content} />
                            </div>
                            {message.role === 'assistant' && message.stats && (
                              <div className="chat-message-stats">
                                <span>Latenza: {message.stats.latencyMs} ms</span>
                                {message.stats.usage && (
                                  <span>
                                    Token P/C/T: {message.stats.usage.promptTokens}/
                                    {message.stats.usage.completionTokens}/
                                    {message.stats.usage.totalTokens}
                                  </span>
                                )}
                              </div>
                            )}
                          </div>
                        ))
                      )}
                      {chatLoading && (
                        <div className="chat-message assistant">
                          <div className="chat-role">Assistente</div>
                          <div className="chat-content">Sto generando la risposta...</div>
                        </div>
                      )}
                    </div>

                    <div className="chat-input-row">
                      <input
                        type="text"
                        value={chatInput}
                        onChange={(e) => setChatInput(e.target.value)}
                        placeholder="Scrivi un messaggio..."
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            handleSendMessage();
                          }
                        }}
                        disabled={chatLoading}
                      />
                      <button
                        onClick={handleSendMessage}
                        className="btn btn-start"
                        disabled={chatLoading || !chatInput.trim()}
                      >
                        {chatLoading ? 'Invio...' : 'Invia'}
                      </button>
                      <button
                        onClick={() => setChatMessages([])}
                        className="btn btn-stop"
                        disabled={chatLoading || chatMessages.length === 0}
                      >
                        Pulisci
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
            </React.Fragment>
          </div>
        </section>
      </div>

      {/* Modal per salvare configurazione */}
      {showConfigModal && (
        <div className="modal-overlay" onClick={() => setShowConfigModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h3>💾 Salva Configurazione</h3>
            <div className="form-group">
              <label>Nome file:</label>
              <input
                type="text"
                value={configFilename}
                onChange={(e) => setConfigFilename(e.target.value)}
                placeholder="nome-configurazione"
                onKeyPress={(e) => {
                  if (e.key === 'Enter') {
                    handleSaveConfig();
                  }
                }}
              />
              <small>Il file verrà salvato come .json</small>
            </div>
            <div className="modal-actions">
              <button
                onClick={handleSaveConfig}
                className="btn btn-start"
              >
                💾 Salva
              </button>
              <button
                onClick={() => {
                  setShowConfigModal(false);
                  setConfigFilename('');
                }}
                className="btn btn-stop"
              >
                Annulla
              </button>
            </div>
          </div>
        </div>
      )}
      
      {/* Footer */}
      <footer className="main-footer">
        <strong>Copyright &copy; 2024 <a href="#">vLLM Control Panel</a>.</strong>
        All rights reserved.
        <div className="float-right d-none d-sm-inline-block">
          <b>Version</b> 1.0.0
        </div>
      </footer>
    </div>
  );
}

export default App;
