import React, { useState, useEffect } from 'react'
import { CSVLink } from 'react-csv'
import CausalDAG from './CausalDAG'
import SHAPHeatmap from './SHAPHeatmap'
import PatientFlowSankey from './PatientFlowSankey'
import { getPatientFlow } from '../services/api'

export default function BottleneckReport({ bottlenecks, metrics }) {
  const [selectedBottleneck, setSelectedBottleneck] = useState(null)
  const [flowData, setFlowData] = useState(null)
  const [loadingFlow, setLoadingFlow] = useState(false)
  
  const handleExportPDF = () => {
    window.print()
  }
  
  // Load flow data when a bottleneck is selected
  useEffect(() => {
    if (selectedBottleneck) {
      setLoadingFlow(true)
      getPatientFlow(24, selectedBottleneck.stage)
        .then(data => {
          setFlowData(data)
          setLoadingFlow(false)
        })
        .catch(err => {
          console.error('Failed to load flow data:', err)
          setLoadingFlow(false)
        })
    } else {
      setFlowData(null)
    }
  }, [selectedBottleneck])

  const exportToCSV = () => {
    if (!bottlenecks || bottlenecks.length === 0) return []
    
    return bottlenecks.map(b => ({
      'Bottleneck Name': b.bottleneck_name || 'Unknown',
      Stage: b.stage || 'Unknown',
      'Wait Time (min)': b.current_wait_time_minutes?.toFixed(1) || 0,
      'Impact Score': `${((b.impact_score || 0) * 100).toFixed(0)}%`,
      Severity: b.severity || 'medium',
      Causes: b.causes?.join('; ') || '',
      Recommendations: b.recommendations?.join('; ') || ''
    }))
  }
  if (!bottlenecks || bottlenecks.length === 0) {
    return (
      <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-green-900 dark:text-green-200 mb-2">
          ✅ No Significant Bottlenecks
        </h3>
        <p className="text-green-800 dark:text-green-200">
          Your ED is operating efficiently with no major bottlenecks detected.
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Export Buttons */}
      <div className="flex gap-2 justify-end">
        <CSVLink
          data={exportToCSV()}
          filename={`bottlenecks-${new Date().toISOString().split('T')[0]}.csv`}
          className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 text-sm font-medium"
        >
          📥 Export CSV
        </CSVLink>
        <button
          onClick={handleExportPDF}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm font-medium"
        >
          📄 Export PDF
        </button>
      </div>

      <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-6">
        <h3 className="text-xl font-bold text-blue-900 dark:text-blue-200 mb-3">
          🔍 Current Bottlenecks Analysis
        </h3>
        {metrics && (
          <div className="mb-4 grid grid-cols-3 gap-4 text-sm">
            <div>
              <div className="text-gray-600 dark:text-gray-400">Door-to-Doctor</div>
              <div className="text-lg font-bold text-gray-900 dark:text-white">
                {metrics.dtd?.toFixed(1) || 'N/A'} min
              </div>
            </div>
            <div>
              <div className="text-gray-600 dark:text-gray-400">Length of Stay</div>
              <div className="text-lg font-bold text-gray-900 dark:text-white">
                {metrics.los?.toFixed(1) || 'N/A'} min
              </div>
            </div>
            <div>
              <div className="text-gray-600 dark:text-gray-400">LWBS Rate</div>
              <div className="text-lg font-bold text-gray-900 dark:text-white">
                {(metrics.lwbs * 100)?.toFixed(1) || 'N/A'}%
              </div>
            </div>
          </div>
        )}
      </div>

      {bottlenecks.map((bottleneck, idx) => {
        const severityColors = {
          critical: 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800 text-red-900 dark:text-red-200',
          high: 'bg-orange-50 dark:bg-orange-900/20 border-orange-200 dark:border-orange-800 text-orange-900 dark:text-orange-200',
          medium: 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800 text-yellow-900 dark:text-yellow-200',
          low: 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800 text-blue-900 dark:text-blue-200'
        }
        
        const severity = bottleneck.severity || 'medium'
        const colorClass = severityColors[severity] || severityColors.medium
        const [showCausalViz, setShowCausalViz] = useState(false)
        const [showFlowViz, setShowFlowViz] = useState(false)
        const causalData = bottleneck.metadata?.causal_analysis || null
        const isSelected = selectedBottleneck?.bottleneck_name === bottleneck.bottleneck_name
        const flowCascade = bottleneck.flow_cascade || {}
        
        return (
          <div key={idx} className={`border rounded-lg p-6 ${colorClass} ${isSelected ? 'ring-2 ring-purple-500' : ''}`}>
            <div className="flex items-start justify-between mb-3">
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <h4 className="text-lg font-semibold mb-1">
                    {bottleneck.bottleneck_name || `Bottleneck ${idx + 1}`}
                  </h4>
                  {isSelected && (
                    <span className="text-xs bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 px-2 py-1 rounded">
                      Selected
                    </span>
                  )}
                </div>
                <p className="text-sm opacity-75">
                  Stage: {bottleneck.stage || 'Unknown'}
                </p>
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold">
                  {bottleneck.current_wait_time_minutes?.toFixed(0) || 'N/A'}
                </div>
                <div className="text-xs opacity-75">minutes wait</div>
              </div>
            </div>
            
            <div className="mb-3">
              <div className="text-sm font-medium mb-1">Impact Score: {((bottleneck.impact_score || 0) * 100).toFixed(0)}%</div>
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                <div 
                  className="bg-current h-2 rounded-full"
                  style={{ width: `${(bottleneck.impact_score || 0) * 100}%` }}
                />
              </div>
            </div>
            
            {bottleneck.causes && bottleneck.causes.length > 0 && (
              <div className="mb-3">
                <div className="text-sm font-medium mb-1">Causes:</div>
                <ul className="list-disc list-inside text-sm space-y-1">
                  {bottleneck.causes.map((cause, cIdx) => (
                    <li key={cIdx}>{cause}</li>
                  ))}
                </ul>
              </div>
            )}
            
            {/* Drill-Down Button */}
            <div className="mb-3">
              <button
                onClick={() => {
                  if (isSelected) {
                    setSelectedBottleneck(null)
                    setShowFlowViz(false)
                  } else {
                    setSelectedBottleneck(bottleneck)
                    setShowFlowViz(true)
                  }
                }}
                className={`text-sm px-3 py-1 rounded transition-colors ${
                  isSelected
                    ? 'bg-purple-600 text-white hover:bg-purple-700'
                    : 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 hover:bg-blue-200 dark:hover:bg-blue-900/50'
                }`}
              >
                {isSelected ? '▼' : '▶'} {isSelected ? 'Hide' : 'Show'} Patient Flow Cascade
              </button>
            </div>
            
            {/* Patient Flow Cascade */}
            {isSelected && showFlowViz && (
              <div className="mb-3">
                {bottleneck.flow_cascade && bottleneck.flow_cascade.cascade_paths && bottleneck.flow_cascade.cascade_paths.length > 0 ? (
                  <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg border border-purple-200 dark:border-purple-800">
                    <h4 className="font-semibold text-purple-900 dark:text-purple-200 mb-3">Patient Flow Cascade:</h4>
                    
                    {/* Cascade Paths */}
                    <div className="space-y-2 mb-4">
                      {bottleneck.flow_cascade.cascade_paths.map((path, idx) => (
                        <div key={idx} className="bg-white dark:bg-gray-800 p-3 rounded border border-purple-200 dark:border-purple-700">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="text-lg">→</span>
                            <span className="font-medium text-gray-900 dark:text-white">
                              {path.from} → {path.to}
                            </span>
                            <span className="text-xs text-gray-500 dark:text-gray-400">
                              (Strength: {path.strength})
                            </span>
                          </div>
                          <div className="text-sm text-gray-700 dark:text-gray-300 ml-6">
                            {path.delay_minutes} min delay | {path.affected_patients} patients affected
                          </div>
                          {bottleneck.flow_cascade.downstream_impacts?.[path.to] && (
                            <div className="text-xs text-gray-600 dark:text-gray-400 ml-6 mt-1">
                              {bottleneck.flow_cascade.downstream_impacts[path.to].impact_description}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>

                    {/* System Impact */}
                    {bottleneck.flow_cascade.system_impact && (
                      <div className="mt-3 pt-3 border-t border-purple-200 dark:border-purple-700">
                        <div className="text-sm text-gray-700 dark:text-gray-300">
                          <strong>System Impact:</strong> {bottleneck.flow_cascade.system_impact.total_patients_affected} patients affected, 
                          {bottleneck.flow_cascade.system_impact.total_delay_minutes} min total delay, 
                          {bottleneck.flow_cascade.system_impact.cascade_depth} cascade paths ({bottleneck.flow_cascade.system_impact.severity} severity)
                        </div>
                      </div>
                    )}

                    {/* Upstream Dependencies */}
                    {bottleneck.flow_cascade.upstream_dependencies && bottleneck.flow_cascade.upstream_dependencies.length > 0 && (
                      <div className="mt-3 pt-3 border-t border-purple-200 dark:border-purple-700">
                        <div className="text-sm font-medium text-gray-900 dark:text-white mb-2">Upstream Dependencies:</div>
                        <div className="space-y-1">
                          {bottleneck.flow_cascade.upstream_dependencies.map((dep, idx) => (
                            <div key={idx} className="text-xs text-gray-600 dark:text-gray-400">
                              • {dep.description} (strength: {dep.dependency_strength})
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Optional: Sankey Visualization */}
                    {loadingFlow ? (
                      <div className="mt-4">
                        <p className="text-gray-500 dark:text-gray-400 text-sm">Loading detailed flow visualization...</p>
                      </div>
                    ) : flowData ? (
                      <div className="mt-4">
                        <PatientFlowSankey 
                          flowData={flowData} 
                          bottleneckStage={bottleneck.stage}
                        />
                      </div>
                    ) : null}
                  </div>
                ) : (
                  <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
                    <p className="text-gray-500 dark:text-gray-400 text-sm">
                      {loadingFlow ? 'Loading patient flow data...' : 'No patient flow cascade data available.'}
                    </p>
                  </div>
                )}
              </div>
            )}
            
            {/* Temporal Analysis Display */}
            {bottleneck.metadata?.temporal_analysis && (
              <div className="mb-3 p-3 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
                <div className="text-sm font-semibold text-blue-900 dark:text-blue-200 mb-1">
                  ⏰ Temporal Pattern
                </div>
                <div className="text-xs text-blue-800 dark:text-blue-300">
                  {bottleneck.metadata.temporal_analysis.pattern || 'No clear pattern'}
                  {bottleneck.metadata.temporal_analysis.peak_range && (
                    <span className="ml-2 font-medium">
                      (Peak: {bottleneck.metadata.temporal_analysis.peak_range})
                    </span>
                  )}
                </div>
              </div>
            )}
            
            {/* Causal Visualization Toggle */}
            {causalData && (
              <div className="mb-3">
                <button
                  onClick={() => setShowCausalViz(!showCausalViz)}
                  className="text-sm px-3 py-1 bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 rounded hover:bg-purple-200 dark:hover:bg-purple-900/50 transition-colors"
                >
                  {showCausalViz ? '▼' : '▶'} Causal Analysis DAG
                </button>
                {showCausalViz && (
                  <div className="mt-3">
                    <CausalDAG 
                      causalData={causalData} 
                      bottleneckName={bottleneck.bottleneck_name}
                    />
                    {/* Counterfactual Display */}
                    {causalData.counterfactuals && causalData.counterfactuals.length > 0 && (
                      <div className="mt-4 p-4 bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded-lg">
                        <h5 className="text-sm font-semibold text-purple-900 dark:text-purple-200 mb-2">
                          Counterfactual Insights
                        </h5>
                        {causalData.counterfactuals.slice(0, 2).map((cf, cfIdx) => (
                          <div key={cfIdx} className="text-xs text-purple-800 dark:text-purple-300 mb-2">
                            <strong>{cf.scenario || 'Intervention'}:</strong> {cf.expected_outcome || 'N/A'}
                            {cf.confidence_interval && (
                              <span className="text-gray-600 dark:text-gray-400 ml-2">
                                (CI: {cf.confidence_interval[0]?.toFixed(1)}-{cf.confidence_interval[1]?.toFixed(1)})
                              </span>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                    
                    {/* SHAP Heatmap */}
                    {causalData.feature_attributions && (
                      <div className="mt-4">
                        <SHAPHeatmap 
                          shapData={causalData.feature_attributions}
                          featureNames={Object.keys(causalData.feature_attributions.attributions || {})}
                        />
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
            
            {bottleneck.recommendations && bottleneck.recommendations.length > 0 && (
              <div>
                <div className="text-sm font-medium mb-1">Recommendations:</div>
                <ul className="list-disc list-inside text-sm space-y-1">
                  {bottleneck.recommendations.map((rec, rIdx) => (
                    <li key={rIdx}>{rec}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

