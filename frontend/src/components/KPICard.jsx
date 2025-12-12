export default function KPICard({ title, value, threshold, current, trend }) {
  const isAboveThreshold = current > threshold
  const trendIcon = trend === 'down' ? '↓' : trend === 'up' ? '↑' : '→'
  const trendColor = trend === 'down' ? 'text-green-600' : trend === 'up' ? 'text-red-600' : 'text-gray-600'

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg shadow p-6 ${isAboveThreshold ? 'border-l-4 border-red-500' : ''}`}>
      <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-1">{title}</h3>
      <div className="flex items-baseline justify-between">
        <p className="text-2xl font-bold text-gray-900 dark:text-white">{value}</p>
        <span className={`text-sm ${trendColor}`}>{trendIcon}</span>
      </div>
      {isAboveThreshold && (
        <p className="text-xs text-red-600 dark:text-red-400 mt-2">Above threshold ({threshold})</p>
      )}
    </div>
  )
}

