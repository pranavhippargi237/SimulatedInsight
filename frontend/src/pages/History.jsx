import { useState, useEffect } from 'react'
import { CSVLink } from 'react-csv'

export default function History() {
  const [history, setHistory] = useState([])

  useEffect(() => {
    // Load from localStorage
    const saved = localStorage.getItem('simulation_history')
    if (saved) {
      setHistory(JSON.parse(saved))
    }
  }, [])

  const clearHistory = () => {
    localStorage.removeItem('simulation_history')
    setHistory([])
  }

  if (history.length === 0) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">Simulation History</h2>
        <p className="text-gray-600 dark:text-gray-400">No simulation history yet.</p>
      </div>
    )
  }

  const csvData = history.map((item, idx) => ({
    id: idx + 1,
    date: new Date(item.timestamp).toLocaleString(),
    query: item.query,
    dtd_change: item.result?.deltas?.dtd_change || 0,
    los_change: item.result?.deltas?.los_change || 0,
    lwbs_drop: item.result?.deltas?.lwbs_drop || 0,
    confidence: item.result?.confidence || 0,
  }))

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Simulation History</h2>
        <div className="space-x-2">
          <CSVLink
            data={csvData}
            filename="simulation_history.csv"
            className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700"
          >
            Export CSV
          </CSVLink>
          <button
            onClick={clearHistory}
            className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
          >
            Clear History
          </button>
        </div>
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-lg shadow overflow-hidden">
        <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
          <thead className="bg-gray-50 dark:bg-gray-700">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                Date
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                Query
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                DTD Change
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                LOS Change
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                LWBS Drop
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                Confidence
              </th>
            </tr>
          </thead>
          <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
            {history.map((item, idx) => (
              <tr key={idx}>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                  {new Date(item.timestamp).toLocaleString()}
                </td>
                <td className="px-6 py-4 text-sm text-gray-900 dark:text-white">
                  {item.query}
                </td>
                <td className={`px-6 py-4 whitespace-nowrap text-sm ${
                  (item.result?.deltas?.dtd_change || 0) < 0 ? 'text-green-600' : 'text-red-600'
                }`}>
                  {(item.result?.deltas?.dtd_change || 0).toFixed(1)}%
                </td>
                <td className={`px-6 py-4 whitespace-nowrap text-sm ${
                  (item.result?.deltas?.los_change || 0) < 0 ? 'text-green-600' : 'text-red-600'
                }`}>
                  {(item.result?.deltas?.los_change || 0).toFixed(1)}%
                </td>
                <td className={`px-6 py-4 whitespace-nowrap text-sm ${
                  (item.result?.deltas?.lwbs_drop || 0) < 0 ? 'text-green-600' : 'text-red-600'
                }`}>
                  {(item.result?.deltas?.lwbs_drop || 0).toFixed(1)}%
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                  {((item.result?.confidence || 0) * 100).toFixed(1)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

