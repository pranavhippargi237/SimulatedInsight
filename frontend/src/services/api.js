import axios from 'axios'

// Ensure API_URL always includes /api prefix
const envUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'
const API_URL = envUrl.endsWith('/api') ? envUrl : `${envUrl}/api`

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 second timeout for all requests
})

// Add response interceptor to handle errors globally
api.interceptors.response.use(
  (response) => response,
  (error) => {
    // Enhance network errors with more context
    if (error.code === 'ERR_NETWORK' || error.code === 'ERR_CONNECTION_RESET') {
      error.message = 'Network Error: Unable to connect to backend server. Please ensure the server is running.'
    }
    return Promise.reject(error)
  }
)

export const ingestCSV = async (file, resetFirst = true) => {
  const formData = new FormData()
  formData.append('file', file)
  // By default, reset existing data before uploading to ensure metrics reflect only the new CSV
  const response = await api.post(`/ingest/csv?reset_first=${resetFirst}`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
  return response.data
}

export const ingestJSON = async (events) => {
  const response = await api.post('/ingest/json', events)
  return response.data
}

export const getMetrics = async (window = '24h', includeAnomalies = true) => {
  // Add cache-busting timestamp to ensure fresh data
  const response = await api.get('/metrics', {
    params: { 
      window, 
      include_anomalies: includeAnomalies,
      _t: Date.now() // Cache buster
    },
  })
  return response.data
}

export const detectBottlenecks = async (windowHours = 24, topN = 3) => {
  // Add cache-busting timestamp to ensure fresh data
  const response = await api.post('/detect', null, {
    params: { 
      window_hours: windowHours, 
      top_n: topN,
      _t: Date.now() // Cache buster
    },
    timeout: 45000, // 45 second timeout for bottleneck detection (increased from 30s)
  })
  return response.data
}

export const runSimulation = async (scenario) => {
  const response = await api.post('/simulate', scenario)
  return response.data
}

export const runSimulationNLP = async (query) => {
  const response = await api.post('/simulate/nlp', { query })
  return response.data
}

export const optimize = async (bottlenecks, constraints, objective = 'minimize_dtd') => {
  const response = await api.post('/optimize', {
    bottlenecks,
    constraints,
    objective,
  })
  return response.data
}

export const healthCheck = async () => {
  const response = await api.get('/health')
  return response.data
}

export const askAdvisor = async (query) => {
  const response = await axios.post(`${API_URL}/advisor/ask?query=${encodeURIComponent(query)}`)
  return response.data
}

export const processIntelligentQuery = async (query) => {
  const response = await axios.post(`${API_URL}/intelligent/query`, { query })
  return response.data
}

export const chat = async (query, conversationId = 'default') => {
  const response = await axios.post(`${API_URL}/chat`, { query }, {
    params: { conversation_id: conversationId }
  })
  return response.data
}

export const clearConversation = async (conversationId = 'default') => {
  const response = await axios.post(`${API_URL}/chat/clear`, null, {
    params: { conversation_id: conversationId }
  })
  return response.data
}

export const getPatientFlow = async (windowHours = 24, stageFilter = null) => {
  const params = { window_hours: windowHours }
  if (stageFilter) {
    params.stage_filter = stageFilter
  }
  const response = await api.get('/flow/sankey', { params })
  return response.data
}

export default api

