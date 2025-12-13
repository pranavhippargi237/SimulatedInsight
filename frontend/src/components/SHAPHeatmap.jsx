import React from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts'

export default function SHAPHeatmap({ shapData, featureNames }) {
  if (!shapData || !shapData.attributions || Object.keys(shapData.attributions).length === 0) {
    return (
      <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-8 text-center">
        <p className="text-gray-500 dark:text-gray-400">
          No SHAP attribution data available.
        </p>
      </div>
    )
  }

  // Prepare data for visualization
  const data = Object.entries(shapData.attributions)
    .map(([feature, value]) => ({
      feature: feature.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
      value: Math.abs(value),
      rawValue: value,
      isPositive: value > 0
    }))
    .sort((a, b) => b.value - a.value)

  // Color scale: red for positive (increases wait), green for negative (decreases wait)
  const getColor = (isPositive, value) => {
    const intensity = Math.min(value / 50, 1) // Normalize to 0-1
    if (isPositive) {
      // Red scale (increases wait time)
      const r = 239 + Math.floor((255 - 239) * (1 - intensity))
      const g = 68 + Math.floor((68 - 68) * (1 - intensity))
      const b = 68 + Math.floor((68 - 68) * (1 - intensity))
      return `rgb(${r}, ${g}, ${b})`
    } else {
      // Green scale (decreases wait time)
      const r = 34 + Math.floor((34 - 34) * (1 - intensity))
      const g = 197 + Math.floor((255 - 197) * (1 - intensity))
      const b = 94 + Math.floor((94 - 94) * (1 - intensity))
      return `rgb(${r}, ${g}, ${b})`
    }
  }

  return (
    <div className="w-full bg-white dark:bg-gray-900 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
      <div className="mb-4">
        <h4 className="text-sm font-semibold text-gray-900 dark:text-gray-100 mb-2">
          SHAP Feature Attribution Heatmap
        </h4>
        <p className="text-xs text-gray-600 dark:text-gray-400 mb-3">
          Red bars = increases wait time | Green bars = decreases wait time
        </p>
      </div>
      
      <ResponsiveContainer width="100%" height={Math.max(300, data.length * 40)}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis 
            type="number" 
            label={{ value: 'Contribution (%)', position: 'insideBottom', offset: -5 }}
            stroke="#6b7280"
          />
          <YAxis 
            dataKey="feature" 
            type="category" 
            width={90}
            stroke="#6b7280"
            tick={{ fontSize: 11 }}
          />
          <Tooltip
            formatter={(value, name) => [`${value.toFixed(1)}%`, 'Contribution']}
            contentStyle={{ backgroundColor: '#f9fafb', border: '1px solid #e5e7eb', borderRadius: '4px' }}
          />
          <Legend />
          <Bar dataKey="value" name="SHAP Contribution">
            {data.map((entry, index) => (
              <Cell 
                key={`cell-${index}`} 
                fill={getColor(entry.isPositive, entry.value)}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      
      <div className="mt-4 text-xs text-gray-600 dark:text-gray-400">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-red-500 rounded"></div>
            <span>Increases wait time</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-green-500 rounded"></div>
            <span>Decreases wait time</span>
          </div>
        </div>
      </div>
    </div>
  )
}

