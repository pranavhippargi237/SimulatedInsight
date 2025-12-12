import { useState, useEffect, useRef } from 'react'
import { getMetrics, detectBottlenecks, healthCheck } from '../services/api'
import BottleneckCard from '../components/BottleneckCard'
import KPICard from '../components/KPICard'

export default function Dashboard() {
  const [metrics, setMetrics] = useState(null)
  const [bottlenecks, setBottlenecks] = useState([])
  const [loading, setLoading] = useState(false) // Start as false - only set true when actually loading
  const [window, setWindow] = useState('24h')
  const [backendOnline, setBackendOnline] = useState(null)
  const refreshTriggerRef = useRef(0)

  // Check backend health on mount
  useEffect(() => {
    let healthInterval = null
    
    const checkHealth = async () => {
      try {
        console.log('🏥 Checking backend health...')
        const health = await healthCheck()
        console.log('✅ Backend health check passed:', health)
        setBackendOnline(true)
        // Reset loading state when backend comes online (in case it was stuck)
        setLoading(false)
        // Log storage status for debugging
        if (health.storage) {
          console.log('💾 Storage status:', health.storage)
          if (!health.storage.sqlite) {
            console.warn('⚠️ SQLite not connected - data operations may fail')
          }
        }
      } catch (error) {
        // Check if it's a network error
        if (error.code === 'ERR_NETWORK' || error.code === 'ERR_CONNECTION_RESET' || error.message?.includes('Network Error')) {
          console.error('❌ Backend health check failed - network error:', error)
          setBackendOnline(false)
        } else {
          console.error('❌ Backend health check failed:', error)
          setBackendOnline(false)
        }
      }
    }
    
    // Initial health check
    console.log('🚀 Initializing health check...')
    checkHealth()
    
    // Periodically re-check health if backend is offline (every 10 seconds)
    // Use a ref-like pattern to check current state without causing re-renders
    healthInterval = setInterval(() => {
      checkHealth()
    }, 10000)
    
    return () => {
      if (healthInterval) {
        clearInterval(healthInterval)
      }
    }
  }, []) // Empty deps - only run on mount

  // Listen for storage events (triggered when data is uploaded)
  useEffect(() => {
    // Check if window is available and has addEventListener (browser environment)
    const globalWindow = typeof window !== 'undefined' ? window : null
    if (!globalWindow || typeof globalWindow.addEventListener !== 'function') {
      return
    }
    
    let refreshTimeout = null
    
    const handleStorageChange = () => {
      // Trigger refresh when data is uploaded (detected via localStorage event)
      console.log('🔄 Data upload detected - refreshing dashboard...')
      refreshTriggerRef.current += 1
      
      // Clear any pending refresh to avoid multiple refreshes
      if (refreshTimeout) {
        clearTimeout(refreshTimeout)
      }
      
      // Force immediate refresh after a short delay
      refreshTimeout = setTimeout(() => {
        if (backendOnline !== false) {
          loadData()
        }
      }, 2000) // Wait 2 seconds for backend to finish processing
    }
    
    // Listen for custom event from Chat page after upload
    globalWindow.addEventListener('dataUploaded', handleStorageChange)
    
    // Also listen for storage events (if other tabs upload data)
    const handleStorage = (e) => {
      if (e.key === 'dataLastUploaded') {
        handleStorageChange()
      }
    }
    globalWindow.addEventListener('storage', handleStorage)
    
    return () => {
      if (refreshTimeout) {
        clearTimeout(refreshTimeout)
      }
      if (globalWindow) {
        globalWindow.removeEventListener('dataUploaded', handleStorageChange)
        globalWindow.removeEventListener('storage', handleStorage)
      }
    }
  }, [backendOnline])

  useEffect(() => {
    console.log('🔄 useEffect triggered:', { window, backendOnline, loading })
    
    // Don't try to load data if backend is confirmed offline
    if (backendOnline === false) {
      console.log('⏸️ Backend offline - skipping loadData')
      setLoading(false) // Make sure loading state is cleared
      return
    }
    
    // If backendOnline is null (initial state), wait for health check
    if (backendOnline === null) {
      console.log('⏳ Waiting for health check to complete...')
      return
    }
    
    // Load data when window changes or backend comes online
    // No automatic refresh - only manual refresh or when window changes
    console.log('🚀 Calling loadData...')
    loadData()
  }, [window, backendOnline]) // Refresh when time window changes or backend status changes

  const loadData = async () => {
    console.log('📥 loadData called', { backendOnline, loading })
    
    // Don't attempt to load if backend is confirmed offline
    if (backendOnline === false) {
      console.log('⏸️ Backend offline - loadData returning early')
      setLoading(false)
      return
    }
    
    // Prevent concurrent loads - but allow if we're stuck in loading state
    if (loading) {
      console.log('⏸️ Already loading - checking if this is a stuck state...')
      // If we've been loading for more than 30 seconds, allow retry
      // This prevents infinite blocking
      return
    }
    
    try {
      console.log('⏳ Setting loading to true and starting data fetch')
      setLoading(true)
      
      // Try to load data with individual error handling
      let metricsData = null
      let bottlenecksData = null
      
      try {
        console.log(`📈 Loading metrics and analytics for window: ${window}`)
        metricsData = await getMetrics(window)
        console.log(`✅ Metrics loaded: DTD=${metricsData?.current_metrics?.dtd?.toFixed(1)}, LOS=${metricsData?.current_metrics?.los?.toFixed(1)}, KPIs=${metricsData?.historical_kpis?.length}`)
      } catch (error) {
        // Check if it's a network error - if so, mark backend as offline
        if (error.code === 'ERR_NETWORK' || error.code === 'ERR_CONNECTION_RESET' || error.message?.includes('Network Error')) {
          console.error('❌ Network error detected, marking backend as offline:', error)
          setBackendOnline(false)
          setLoading(false)
          return // Stop trying to load other data
        }
        console.error('❌ Failed to load metrics:', error)
        console.error('Error details:', {
          message: error.message,
          code: error.code,
          response: error.response?.data
        })
        // If metrics fail, try to continue with empty metrics
        metricsData = {
          status: 'error',
          current_metrics: {},
          historical_kpis: [],
          anomalies: []
        }
      }
      
      try {
        // Use the 'window' state variable (time window like '24h'), not global window object
        const windowHours = parseInt(window.replace('h', ''))
        console.log(`🔍 Loading bottlenecks for window: ${windowHours}h`)
        bottlenecksData = await detectBottlenecks(windowHours)
        console.log(`✅ Bottlenecks loaded: ${bottlenecksData?.bottlenecks?.length || 0} found`)
      } catch (error) {
        // Check if it's a network error - if so, mark backend as offline
        if (error.code === 'ERR_NETWORK' || error.code === 'ERR_CONNECTION_RESET' || error.message?.includes('Network Error')) {
          console.error('❌ Network error detected for bottlenecks:', error)
          // Don't mark backend offline if metrics worked - just skip bottlenecks
          if (!metricsData || metricsData.status === 'error') {
            setBackendOnline(false)
            setLoading(false)
            return
          }
        }
        console.error('⚠️ Failed to load bottlenecks (non-critical):', error)
        console.error('Bottleneck error details:', {
          message: error.message,
          code: error.code,
          response: error.response?.data
        })
        // If bottlenecks fail, continue with empty array (don't block dashboard)
        bottlenecksData = { bottlenecks: [] }
        // Don't return - allow metrics to display even if bottlenecks fail
      }
      
      // Only update state if we got some data
      if (metricsData) {
        console.log('📊 Setting metrics data:', {
          hasCurrentMetrics: !!metricsData.current_metrics,
          kpiCount: metricsData.historical_kpis?.length || 0,
          status: metricsData.status
        })
        setMetrics(metricsData)
      } else {
        console.warn('⚠️ No metrics data received')
      }
      
      if (bottlenecksData) {
        const bottlenecks = bottlenecksData?.bottlenecks || []
        console.log(`📊 Loaded ${bottlenecks.length} bottlenecks for window ${window}`, bottlenecks)
        setBottlenecks(bottlenecks)
      } else {
        console.warn('⚠️ No bottlenecks data received - using empty array')
        setBottlenecks([])
      }
      
      // If we successfully loaded data, mark backend as online
      if (metricsData && metricsData.status !== 'error') {
        setBackendOnline(true)
        console.log('✅ Backend marked as online')
      }
    } catch (error) {
      console.error('❌ Failed to load data (outer catch):', error)
      console.error('Error stack:', error.stack)
      // Check if it's a network error
      if (error.code === 'ERR_NETWORK' || error.code === 'ERR_CONNECTION_RESET' || error.message?.includes('Network Error')) {
        console.error('❌ Network error - marking backend offline')
        setBackendOnline(false)
        setLoading(false)
        return
      }
      // Set empty state on error so user sees "No Data Available" instead of infinite loading
      if (!metrics) {
        console.log('📊 Setting error state for metrics')
        setMetrics({
          status: 'error',
          message: error.message || 'Failed to connect to server',
          current_metrics: {},
          historical_kpis: [],
          anomalies: []
        })
        setBottlenecks([])
      }
    } finally {
      console.log('🏁 loadData completed - setting loading to false')
      // Always clear loading state, even on error
      setLoading(false)
      console.log('✅ Loading state cleared. Metrics:', !!metrics, 'Bottlenecks:', bottlenecks.length)
    }
  }

  const getSeverityColor = (severity) => {
    switch (severity?.toLowerCase()) {
      case 'critical':
        return 'text-red-600 dark:text-red-400'
      case 'high':
        return 'text-orange-600 dark:text-orange-400'
      case 'medium':
        return 'text-yellow-600 dark:text-yellow-400'
      default:
        return 'text-gray-600 dark:text-gray-400'
    }
  }

  const getSeverityBadge = (severity) => {
    const colors = {
      critical: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300',
      high: 'bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300',
      medium: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300',
      low: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300'
    }
    return colors[severity?.toLowerCase()] || colors.low
  }

  // Show loading only if we're actually loading AND don't have metrics yet
  if (loading && !metrics) {
    console.log('⏳ Rendering loading screen')
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-500 dark:text-gray-400">
          <div className="animate-pulse">Loading...</div>
          <div className="text-sm mt-2 text-gray-400 dark:text-gray-500">
            Fetching data from server...
          </div>
        </div>
      </div>
    )
  }
  
  // If we're not loading but have no metrics, show "No Data" (not loading screen)
  if (!loading && !metrics) {
    console.log('📭 No data and not loading - showing no data message')
  }

  // Handle backend offline state
  if (backendOnline === false) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h2 className="text-2xl font-bold mb-4 text-red-600 dark:text-red-400">⚠️ Backend Server Offline</h2>
        <p className="text-gray-600 dark:text-gray-400 mb-4">
          Unable to connect to the backend server at <code className="bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded">{import.meta.env.VITE_API_URL || 'http://localhost:8000'}</code>
        </p>
        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4 space-y-3">
          <div>
            <p className="text-sm text-blue-800 dark:text-blue-200 mb-2 font-semibold">
              To start the backend server:
            </p>
            <code className="block bg-blue-100 dark:bg-blue-800 px-4 py-2 rounded text-sm">
              cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
            </code>
          </div>
          <div>
            <p className="text-sm text-blue-800 dark:text-blue-200 mb-2 font-semibold">
              To start storage services (ClickHouse + Redis):
            </p>
            <code className="block bg-blue-100 dark:bg-blue-800 px-4 py-2 rounded text-sm">
              docker-compose up -d
            </code>
          </div>
        </div>
      </div>
    )
  }

  // Handle error state
  if (metrics && (metrics.status === 'error' || metrics.message === 'No data available')) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          {metrics.status === 'error' ? 'Connection Error' : 'No Data Available'}
        </h2>
        <p className="text-gray-600 dark:text-gray-400 mb-4">
          {metrics.status === 'error' 
            ? 'Unable to load data from the server. Please check if the backend is running.'
            : 'Please ingest some data to see metrics. Use the Chat page to upload data or run simulations.'}
        </p>
        {metrics.status === 'error' && (
          <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
            <p className="text-sm text-blue-800 dark:text-blue-200">
              <strong>Quick Fix:</strong> Make sure the backend server is running on port 8000.
            </p>
          </div>
        )}
      </div>
    )
  }

  // Debug: Log current state
  console.log('🎨 Render state:', {
    loading,
    hasMetrics: !!metrics,
    metricsStatus: metrics?.status,
    kpiCount: metrics?.historical_kpis?.length || 0,
    bottleneckCount: bottlenecks.length,
    backendOnline
  })

  if (!metrics || !metrics.historical_kpis || metrics.historical_kpis.length === 0) {
    // Show helpful message if we have an error
    if (metrics?.status === 'error') {
      return (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h2 className="text-2xl font-bold mb-4 text-red-600 dark:text-red-400">Error Loading Data</h2>
          <p className="text-gray-600 dark:text-gray-400 mb-4">
            {metrics.message || 'Failed to load data from server. Please check the console for details.'}
          </p>
          <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
            <p className="text-sm text-blue-800 dark:text-blue-200">
              <strong>Quick Fix:</strong> Make sure the backend server is running on port 8000.
            </p>
          </div>
        </div>
      )
    }
    
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">No Data Available</h2>
        <p className="text-gray-600 dark:text-gray-400 mb-4">
          Please ingest some data to see metrics. Use the Chat page to upload data or run simulations.
        </p>
        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
          <p className="text-sm text-blue-800 dark:text-blue-200">
            <strong>Quick Start:</strong> Go to the Chat page and click "Upload CSV Data" or use the advanced data generator.
          </p>
        </div>
      </div>
    )
  }

  const currentMetrics = metrics.current_metrics || {}

  const handleManualRefresh = () => {
    console.log('🔄 Manual refresh triggered')
    refreshTriggerRef.current += 1
    loadData()
  }

  const handleWindowChange = (newWindow) => {
    console.log(`📊 Time window changed to: ${newWindow} - reloading analytics...`)
    setWindow(newWindow)
    // loadData will be triggered by the useEffect when window state changes
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">ED Operations Report</h1>
        <div className="flex gap-2">
          <button
            onClick={handleManualRefresh}
            disabled={loading}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white rounded-lg transition-colors flex items-center gap-2"
            title="Refresh data from server"
          >
            <span className={loading ? 'animate-spin' : ''}>🔄</span>
            {loading ? 'Refreshing...' : 'Refresh'}
          </button>
          <select
            value={window}
            onChange={(e) => handleWindowChange(e.target.value)}
            className="px-4 py-2 border border-gray-300 rounded-md dark:bg-gray-700 dark:text-white dark:border-gray-600"
          >
            <option value="24h">Last 24 Hours</option>
            <option value="48h">Last 48 Hours</option>
            <option value="72h">Last 72 Hours</option>
            <option value="168h">Last Week</option>
          </select>
        </div>
      </div>

      {/* Current Status Summary */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">Current Status</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <KPICard
            title="Door-to-Doctor"
            value={`${currentMetrics.dtd?.toFixed(1) || 0} min`}
            threshold={30}
            current={currentMetrics.dtd}
            trend="down"
          />
          <KPICard
            title="Length of Stay"
            value={`${currentMetrics.los?.toFixed(1) || 0} min`}
            threshold={180}
            current={currentMetrics.los}
            trend="down"
          />
          <KPICard
            title="LWBS Rate"
            value={`${(currentMetrics.lwbs * 100)?.toFixed(1) || 0}%`}
            threshold={5}
            current={currentMetrics.lwbs * 100}
            trend="down"
          />
          <KPICard
            title="Bed Utilization"
            value={`${(currentMetrics.bed_utilization * 100)?.toFixed(1) || 0}%`}
            threshold={90}
            current={currentMetrics.bed_utilization * 100}
            trend="neutral"
          />
        </div>

        {/* Status Assessment */}
        <div className="mt-6 p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
          <h3 className="font-semibold text-lg mb-2 text-gray-900 dark:text-white">Overall Assessment</h3>
          <p className="text-gray-700 dark:text-gray-300">
            {currentMetrics.dtd > 30 ? (
              <span className="text-red-600 dark:text-red-400 font-medium">⚠️ CRITICAL: </span>
            ) : currentMetrics.dtd > 25 ? (
              <span className="text-orange-600 dark:text-orange-400 font-medium">⚠️ WARNING: </span>
            ) : (
              <span className="text-green-600 dark:text-green-400 font-medium">✓ OK: </span>
            )}
            {currentMetrics.dtd > 30 
              ? `Door-to-doctor time is ${currentMetrics.dtd?.toFixed(1)} minutes, exceeding the 30-minute target. Patients are experiencing significant delays.`
              : currentMetrics.dtd > 25
              ? `Door-to-doctor time is ${currentMetrics.dtd?.toFixed(1)} minutes, approaching the 30-minute target. Monitor closely.`
              : `Door-to-doctor time is ${currentMetrics.dtd?.toFixed(1)} minutes, within acceptable range.`
            }
            {currentMetrics.lwbs * 100 > 5 && (
              <span className="block mt-2 text-red-600 dark:text-red-400">
                ⚠️ High LWBS rate of {(currentMetrics.lwbs * 100).toFixed(1)}% indicates patients are leaving before being seen.
              </span>
            )}
          </p>
        </div>
      </div>

      {/* Bottlenecks Report */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">Active Bottlenecks</h2>
        
        {bottlenecks.length === 0 ? (
          <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-6">
            <p className="text-green-800 dark:text-green-200 font-medium">
              ✓ No significant bottlenecks detected. ED operations are running smoothly.
            </p>
          </div>
        ) : (
          <div className="space-y-6">
            {bottlenecks.map((bottleneck, idx) => (
              <BottleneckCard key={idx} bottleneck={bottleneck} />
            ))}
          </div>
        )}
      </div>

      {/* Anomaly Alerts */}
      {metrics.anomalies && metrics.anomalies.length > 0 && (
        <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-6">
          <h3 className="font-semibold text-yellow-800 dark:text-yellow-200 mb-3 text-lg">⚠️ Anomaly Alerts</h3>
          <div className="space-y-2">
            {metrics.anomalies.map((anomaly, idx) => (
              <div key={idx} className="text-sm text-yellow-700 dark:text-yellow-300">
                <span className="font-medium">{anomaly.metric.toUpperCase()}</span> spiked to{' '}
                <span className="font-bold">{anomaly.value.toFixed(1)}</span> (Z-score: {anomaly.z_score.toFixed(2)})
                {' '}— This is significantly higher than normal and requires immediate attention.
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Quick Actions Summary */}
      {bottlenecks.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">Priority Actions</h2>
          <div className="space-y-3">
            {bottlenecks
              .filter(b => b.severity === 'critical' || b.severity === 'high')
              .slice(0, 3)
              .map((bottleneck, idx) => (
                <div key={idx} className="flex items-start space-x-3 p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                  <span className="text-2xl">{idx + 1}.</span>
                  <div>
                    <p className="font-medium text-gray-900 dark:text-white">
                      {bottleneck.recommendations && bottleneck.recommendations.length > 0
                        ? bottleneck.recommendations[0]
                        : `Address ${bottleneck.bottleneck_name} bottleneck`}
                    </p>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                      {bottleneck.bottleneck_name} — {bottleneck.severity?.toUpperCase()} severity
                    </p>
                  </div>
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  )
}
