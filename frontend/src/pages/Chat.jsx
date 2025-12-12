import { useState, useRef, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { chat, ingestCSV, clearConversation, getMetrics } from '../services/api'
import SimulationResult from '../components/SimulationResult'
import ActionPlan from '../components/ActionPlan'
import BottleneckReport from '../components/BottleneckReport'
import StructuredAnalysis from '../components/StructuredAnalysis'
import FileUpload from '../components/FileUpload'

export default function Chat() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [simulationResult, setSimulationResult] = useState(null)
  const [actionPlan, setActionPlan] = useState(null)
  const [bottlenecks, setBottlenecks] = useState(null)
  const [currentMetrics, setCurrentMetrics] = useState(null)
  const [deepAnalysis, setDeepAnalysis] = useState(null)
  const [initialLoading, setInitialLoading] = useState(true)
  const [justUploaded, setJustUploaded] = useState(false)
  const [conversationId] = useState(() => {
    // Use stable conversation id across reloads to preserve context
    const stored = localStorage.getItem('conversation_id')
    if (stored) return stored
    const generated = `conv_${crypto.randomUUID ? crypto.randomUUID() : Date.now()}`
    localStorage.setItem('conversation_id', generated)
    return generated
  })
  const messagesEndRef = useRef(null)

  const exampleQueries = [
    "What if we add 2 nurses?",
    "What should I do to reduce wait times?",
    "What are my current bottlenecks?",
    "Compare adding 2 vs 3 doctors",
    "Explain why DTD increased",
  ]

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Load initial metrics and check for data on mount
  useEffect(() => {
    const loadInitialData = async () => {
      try {
        setInitialLoading(true)
        const metrics = await getMetrics(24)
        // Metrics API returns data in current_metrics nested object
        const currentMetricsData = metrics?.current_metrics || metrics
        if (currentMetricsData && (currentMetricsData.dtd || currentMetricsData.los || currentMetricsData.lwbs !== undefined)) {
          setCurrentMetrics(currentMetricsData)
          setJustUploaded(false) // We have data, so clear the upload flag
          // If we have data, automatically ask for analysis
          try {
            const autoQuery = "What are my current bottlenecks?"
            const response = await chat(autoQuery, conversationId)
            const results = response.results || {}
            if (results.bottlenecks) setBottlenecks(results.bottlenecks)
            if (results.current_metrics) setCurrentMetrics(results.current_metrics)
            if (results.deep_analysis) setDeepAnalysis(results.deep_analysis)
          } catch (queryError) {
            console.warn('Auto-query on mount failed:', queryError)
            // Even if query fails, we have metrics, so that's fine
          }
        } else {
          setJustUploaded(false) // No data found, clear the flag
        }
      } catch (error) {
        console.error('Failed to load initial data:', error)
        setJustUploaded(false)
      } finally {
        setInitialLoading(false)
      }
    }
    loadInitialData()
  }, [conversationId])

  const handleSend = async () => {
    if (!input.trim() || loading) return

    const userMessage = { role: 'user', content: input }
    setMessages((prev) => [...prev, userMessage])
    const query = input
    setInput('')
    setLoading(true)

    try {
      // Use conversational AI endpoint
      const response = await chat(query, conversationId)
      
      // Extract results
      const results = response.results || {}
      
      // Set simulation result if available
      if (results.simulation) {
        setSimulationResult(results.simulation)
      } else {
        setSimulationResult(null)
      }
      
      // Set action plan if available
      if (results.action_plan) {
        setActionPlan(results.action_plan)
      } else {
        setActionPlan(null)
      }
      
      // Set bottlenecks if available
      if (results.bottlenecks) {
        setBottlenecks(results.bottlenecks)
      } else {
        setBottlenecks(null)
      }
      
      // Set current metrics if available
      if (results.current_metrics) {
        setCurrentMetrics(results.current_metrics)
      } else {
        setCurrentMetrics(null)
      }
      
      // Set deep analysis if available
      if (results.deep_analysis) {
        setDeepAnalysis(results.deep_analysis)
      } else {
        setDeepAnalysis(null)
      }
      
      // Add assistant response
      const assistantMessage = {
        role: 'assistant',
        content: response.response || "I've processed your query. Here are the results:",
        data: response,
        isActionPlan: !!results.action_plan,
        isFollowUp: response.is_follow_up || false
      }
      setMessages((prev) => [...prev, assistantMessage])
    } catch (error) {
      const errorMessage = {
        role: 'assistant',
        content: `Error: ${error.response?.data?.detail || error.message}`,
        error: true,
      }
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setLoading(false)
    }
  }

  const handleExampleClick = (example) => {
    setInput(example)
  }

  const handleClearConversation = async () => {
    try {
      await clearConversation(conversationId)
      setMessages([])
      setSimulationResult(null)
      setActionPlan(null)
      setBottlenecks(null)
      setCurrentMetrics(null)
      setDeepAnalysis(null)
    } catch (error) {
      console.error('Failed to clear conversation:', error)
    }
  }

  const handleFileUpload = async (file) => {
    try {
      setLoading(true)
      
      // Add timeout for upload
      const uploadPromise = ingestCSV(file, true) // Reset existing data before upload
      const timeoutPromise = new Promise((_, reject) => 
        setTimeout(() => reject(new Error('Upload timeout after 60 seconds')), 60000)
      )
      
      const result = await Promise.race([uploadPromise, timeoutPromise])
      
      // Calculate KPIs after successful ingestion (with timeout)
      let kpiMessage = ''
      try {
        const kpiPromise = fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000/api'}/ingest/calculate-kpis?window_hours=72`, {
          method: 'POST'
        })
        const kpiTimeout = new Promise((_, reject) => 
          setTimeout(() => reject(new Error('KPI calculation timeout')), 30000)
        )
        
        const kpiResult = await Promise.race([kpiPromise, kpiTimeout])
        if (kpiResult.ok) {
          const kpiData = await kpiResult.json()
          kpiMessage = ` Calculated ${kpiData.kpis_calculated || 0} KPI records.`
        }
      } catch (kpiError) {
        console.warn('KPI calculation failed:', kpiError)
        kpiMessage = ' (KPI calculation skipped - you can calculate KPIs later)'
      }
      
      // After upload, reload data and auto-query for bottlenecks
      setJustUploaded(true) // Mark that we just uploaded data
      try {
        // Wait a moment for KPIs to be fully calculated
        await new Promise(resolve => setTimeout(resolve, 2000))
        
        const metrics = await getMetrics(24)
        // Metrics API returns data in current_metrics nested object
        const currentMetricsData = metrics?.current_metrics || metrics
        if (currentMetricsData && (currentMetricsData.dtd || currentMetricsData.los || currentMetricsData.lwbs !== undefined)) {
          setCurrentMetrics(currentMetricsData)
          // Auto-query for bottlenecks after upload
          try {
            const response = await chat("What are my current bottlenecks?", conversationId)
            const results = response.results || {}
            if (results.bottlenecks) setBottlenecks(results.bottlenecks)
            if (results.current_metrics) setCurrentMetrics(results.current_metrics)
            if (results.deep_analysis) setDeepAnalysis(results.deep_analysis)
          } catch (queryError) {
            console.warn('Auto-query after upload failed:', queryError)
            // Even if query fails, we have metrics, so clear the "no data" message
          }
        } else {
          // If metrics are still not available, try again after a longer delay
          setTimeout(async () => {
            const retryMetrics = await getMetrics(24)
            const retryMetricsData = retryMetrics?.current_metrics || retryMetrics
            if (retryMetricsData && (retryMetricsData.dtd || retryMetricsData.los || retryMetricsData.lwbs !== undefined)) {
              setCurrentMetrics(retryMetricsData)
            }
          }, 5000)
        }
      } catch (analysisError) {
        console.warn('Auto-analysis after upload failed:', analysisError)
      }
      
      // Trigger Dashboard refresh after successful upload
      const globalWindow = typeof window !== 'undefined' ? window : null
      if (globalWindow && typeof globalWindow.dispatchEvent === 'function') {
        globalWindow.dispatchEvent(new CustomEvent('dataUploaded'))
        // Also update localStorage to trigger cross-tab refresh
        if (typeof localStorage !== 'undefined') {
          localStorage.setItem('dataLastUploaded', Date.now().toString())
        }
      }
      
      const message = {
        role: 'assistant',
        content: `✅ Successfully ingested ${result.processed} events.${result.invalid > 0 ? ` ${result.invalid} invalid events were skipped.` : ''}${kpiMessage}\n\n📊 Next Steps:\n• Ask me questions naturally (e.g., "What if we add 2 nurses?")\n• I'll understand context and handle follow-up questions\n• Try "What should I do?" for recommendations`,
        hasLink: true
      }
      setMessages((prev) => [...prev, message])
    } catch (error) {
      console.error('Upload error:', error)
      const errorMessage = {
        role: 'assistant',
        content: `❌ Upload failed: ${error.response?.data?.detail || error.message || 'Unknown error'}. Please check the file format and try again.`,
        error: true,
      }
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col h-[calc(100vh-8rem)]">
      <div className="flex-1 overflow-y-auto space-y-4 p-4">
        {messages.length === 0 && (
          <div className="text-center py-8">
            <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
              ED Operations Assistant
            </h2>
            <div className="max-w-2xl mx-auto space-y-4">
              <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4 mb-6">
                <h3 className="font-semibold text-blue-900 dark:text-blue-200 mb-2">💬 Natural Language Interface</h3>
                <p className="text-left text-sm text-blue-800 dark:text-blue-200 mb-3">
                  I'm your conversational ED operations assistant. Talk to me naturally, and I'll understand what you need.
                </p>
                <ul className="text-left text-sm text-blue-800 dark:text-blue-200 space-y-2 list-disc list-inside">
                  <li><strong>Ask questions:</strong> "What if we add 2 nurses?"</li>
                  <li><strong>Follow-up naturally:</strong> "What about on weekends?"</li>
                  <li><strong>Get recommendations:</strong> "What should I do?"</li>
                  <li><strong>Ask for explanations:</strong> "Why did DTD increase?"</li>
                </ul>
              </div>
              <div className="space-y-2">
                <p className="text-sm font-medium text-gray-700 dark:text-gray-300">Try asking:</p>
                {exampleQueries.map((example, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleExampleClick(example)}
                    className="block w-full text-left px-4 py-2 bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 text-sm text-gray-700 dark:text-gray-300"
                  >
                    "{example}"
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-3xl rounded-lg px-4 py-3 ${
                msg.role === 'user'
                  ? 'bg-primary-600 text-white'
                  : msg.error
                  ? 'bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-white'
              }`}
            >
              <div className="whitespace-pre-line">{msg.content}</div>
              {msg.hasLink && (
                <div className="mt-3">
                  <Link
                    to="/"
                    className="inline-flex items-center px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 text-sm font-medium"
                  >
                    📊 View Dashboard →
                  </Link>
                </div>
              )}
              {msg.isFollowUp && (
                <div className="mt-2 text-xs opacity-75 italic">
                  (Understanding this as a follow-up to previous question)
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="bg-gray-100 dark:bg-gray-700 rounded-lg px-4 py-2">
              <p className="text-gray-600 dark:text-gray-400">Thinking...</p>
            </div>
          </div>
        )}

        {simulationResult && (
          <div className="mt-4">
            <SimulationResult result={simulationResult} />
          </div>
        )}

        {actionPlan && (
          <div className="mt-4">
            <ActionPlan plan={actionPlan} />
          </div>
        )}

        {bottlenecks && (
          <div className="mt-4">
            <BottleneckReport bottlenecks={bottlenecks} metrics={currentMetrics} />
          </div>
        )}

        {/* Show data availability message if no data (but not if we just uploaded) */}
        {!initialLoading && !justUploaded && !currentMetrics && !deepAnalysis && !bottlenecks && (
          <div className="mt-4">
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-6">
              <h3 className="text-lg font-semibold text-yellow-900 dark:text-yellow-200 mb-2">
                ⚠️ No Data Available
              </h3>
              <p className="text-yellow-800 dark:text-yellow-300 mb-4">
                No event data found for analysis. Please upload data to generate insights.
              </p>
              <FileUpload onUpload={handleFileUpload} />
            </div>
          </div>
        )}

        {/* Always show deep analysis if available, or show placeholder if bottlenecks exist */}
        {(deepAnalysis || (bottlenecks && bottlenecks.length > 0)) && (
          <div className="mt-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 border border-gray-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
                📊 Deep Analysis: {deepAnalysis?.metric_name || bottlenecks?.[0]?.bottleneck_name || 'ED Operations Analysis'}
              </h3>
              {deepAnalysis ? (
                <StructuredAnalysis 
                  analysis={deepAnalysis} 
                  bottleneck={bottlenecks?.[0] || null}
                />
              ) : (
                <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4">
                  <p className="text-yellow-800 dark:text-yellow-200 text-sm">
                    Deep analysis is being generated... This may take a few seconds. 
                    {bottlenecks && bottlenecks.length > 0 && (
                      <span className="block mt-2">
                        In the meantime, review the {bottlenecks.length} bottleneck{bottlenecks.length > 1 ? 's' : ''} identified above.
                      </span>
                    )}
                  </p>
                </div>
              )}
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <div className="border-t border-gray-200 dark:border-gray-700 p-4 bg-white dark:bg-gray-800">
        <div className="flex items-center justify-between mb-2">
          <FileUpload onUpload={handleFileUpload} />
          {messages.length > 0 && (
            <button
              onClick={handleClearConversation}
              className="px-3 py-1 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200"
            >
              Clear Conversation
            </button>
          )}
        </div>
        <div className="flex space-x-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
            placeholder="Ask me anything about ED operations... (e.g., 'What if we add 2 nurses?')"
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent dark:bg-gray-700 dark:text-white dark:border-gray-600"
            disabled={loading}
          />
          <button
            onClick={handleSend}
            disabled={loading || !input.trim()}
            className="px-6 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  )
}
