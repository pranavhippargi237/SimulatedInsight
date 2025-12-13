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
    } else if (error.code === 'ECONNABORTED' || error.message?.includes('timeout')) {
      // Timeout errors
      const timeoutMsg = error.config?.timeout 
        ? `Request timed out after ${error.config.timeout / 1000} seconds.`
        : 'Request timed out.'
      error.message = `${timeoutMsg} The backend may be slow or unresponsive. Please check if the server is running and try again.`
    }
    return Promise.reject(error)
  }
)

export const ingestCSV = async (file, resetFirst = true) => {
  const startTime = Date.now()
  console.log(`[API] 📤 Starting CSV upload: ${file.name} (${(file.size / 1024).toFixed(2)} KB)`)
  
  const formData = new FormData()
  formData.append('file', file)
  // By default, reset existing data before uploading to ensure metrics reflect only the new CSV
  try {
    const response = await api.post(`/ingest/csv?reset_first=${resetFirst}`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 120000, // 2 minute timeout for file uploads (can be large)
    })
    const duration = Date.now() - startTime
    console.log(`[API] ✅ CSV upload succeeded in ${duration}ms:`, response.data)
    return response.data
  } catch (error) {
    const duration = Date.now() - startTime
    console.error(`[API] ❌ CSV upload failed after ${duration}ms:`, {
      code: error.code,
      message: error.message,
      response: error.response?.data,
      status: error.response?.status
    })
    throw error
  }
}

export const ingestJSON = async (events) => {
  const response = await api.post('/ingest/json', events)
  return response.data
}

export const getMetrics = async (window = '24h', includeAnomalies = true) => {
  const startTime = Date.now()
  console.log(`[API] 📊 Fetching metrics (window: ${window})...`)
  
  // Add cache-busting timestamp to ensure fresh data
  try {
    const response = await api.get('/metrics', {
      params: { 
        window, 
        include_anomalies: includeAnomalies,
        _t: Date.now() // Cache buster
      },
      timeout: 60000, // 60 second timeout for metrics (may take longer with large datasets)
    })
    const duration = Date.now() - startTime
    console.log(`[API] ✅ Metrics fetched in ${duration}ms`)
    return response.data
  } catch (error) {
    const duration = Date.now() - startTime
    console.error(`[API] ❌ Metrics fetch failed after ${duration}ms:`, {
      code: error.code,
      message: error.message,
      response: error.response?.data,
      status: error.response?.status
    })
    throw error
  }
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
  // Health check should be fast - use shorter timeout
  const startTime = Date.now()
  console.log('[API] 🏥 Health check request starting...')
  
  try {
    const response = await api.get('/health', {
      timeout: 5000 // 5 second timeout for health checks
    })
    const duration = Date.now() - startTime
    console.log(`[API] ✅ Health check succeeded in ${duration}ms:`, response.data)
    return response.data
  } catch (error) {
    const duration = Date.now() - startTime
    console.error(`[API] ❌ Health check failed after ${duration}ms:`, {
      code: error.code,
      message: error.message,
      response: error.response?.data,
      status: error.response?.status
    })
    throw error
  }
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
    params: { conversation_id: conversationId },
    timeout: 60000, // 60 second timeout for chat (may take longer for complex queries)
  })
  return response.data
}

export const chatStream = async (query, conversationId = 'default', onChunk, onResults, onDone, onError) => {
  try {
    // Use /v1/chat/stream endpoint
    const baseUrl = API_URL.replace('/api', '/api/v1')
    const url = new URL(`${baseUrl}/chat/stream`)
    url.searchParams.append('conversation_id', conversationId)
    
    const response = await fetch(url.toString(), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query }),
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || '' // Keep incomplete line in buffer

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6))
            
            if (data.type === 'results' && onResults) {
              onResults(data.data)
            } else if (data.type === 'chunk' && onChunk) {
              onChunk(data.content)
            } else if (data.type === 'done' && onDone) {
              onDone()
            } else if (data.type === 'error' && onError) {
              onError(data.message)
            }
          } catch (e) {
            console.error('Failed to parse SSE data:', e, line)
          }
        }
      }
    }
  } catch (error) {
    if (onError) {
      onError(error.message)
    } else {
      throw error
    }
  }
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

