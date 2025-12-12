import React from 'react'
import { CSVLink } from 'react-csv'

export default function ActionPlan({ plan }) {
  const handleExportPDF = () => {
    // Simple PDF export using browser print
    window.print()
  }

  const exportToCSV = () => {
    if (!plan || !plan.recommendations) return []
    
    return plan.recommendations.map(rec => ({
      Priority: rec.priority,
      Action: rec.action,
      Resource: rec.resource_type,
      Quantity: rec.quantity,
      'DTD Reduction': rec.expected_impact?.dtd_reduction || 0,
      'LOS Reduction': rec.expected_impact?.los_reduction || 0,
      'LWBS Drop': rec.expected_impact?.lwbs_drop || 0,
      'ROI %': rec.cost?.roi_percentage || 0,
      'Cost/Year': rec.cost?.per_year || 0,
      'Annual Savings': rec.cost?.annual_savings || 0,
      Confidence: `${((rec.confidence || 0) * 100).toFixed(0)}%`,
      Description: rec.description
    }))
  }
  if (!plan || plan.status === 'no_bottlenecks') {
    return (
      <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-6">
        <h3 className="text-xl font-bold text-green-900 dark:text-green-200 mb-2">
          ✅ No Action Required
        </h3>
        <p className="text-green-800 dark:text-green-200">
          {plan?.message || "No significant bottlenecks detected. Your ED is operating efficiently!"}
        </p>
      </div>
    )
  }

  const { summary, current_state, recommendations, next_steps, forecast } = plan

  return (
    <div className="space-y-6">
      {/* Export Buttons */}
      <div className="flex gap-2 justify-end">
        <CSVLink
          data={exportToCSV()}
          filename={`action-plan-${new Date().toISOString().split('T')[0]}.csv`}
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

      {/* Executive Summary */}
      <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-6">
        <h3 className="text-xl font-bold text-blue-900 dark:text-blue-200 mb-3">
          📋 Executive Summary
        </h3>
        <p className="text-blue-800 dark:text-blue-200 whitespace-pre-line">
          {summary}
        </p>
      </div>

      {/* Current State */}
      {current_state && (
        <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
            Current State
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <div className="text-2xl font-bold text-gray-900 dark:text-white">
                {current_state.total_bottlenecks}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Total Bottlenecks</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-red-600 dark:text-red-400">
                {current_state.critical}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Critical</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">
                {current_state.high}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">High</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-yellow-600 dark:text-yellow-400">
                {current_state.medium}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Medium</div>
            </div>
          </div>
        </div>
      )}

      {/* Forecast */}
      {forecast && forecast.predicted_dtd && (
        <div className="bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-purple-900 dark:text-purple-200 mb-2">
            🔮 Predictive Forecast
          </h3>
          <p className="text-purple-800 dark:text-purple-200">
            If no action is taken, predicted DTD: <strong>{forecast.predicted_dtd.toFixed(1)} min</strong> in the next {forecast.forecast_hours}h 
            (confidence: {((forecast.confidence || 0) * 100).toFixed(0)}%).
          </p>
        </div>
      )}

      {/* Recommendations */}
      {recommendations && recommendations.length > 0 && (
        <div className="space-y-4">
          <h3 className="text-xl font-bold text-gray-900 dark:text-white">
            🎯 Prioritized Recommendations
          </h3>
          {recommendations.map((rec, idx) => (
            <div key={idx} className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-6 shadow-sm">
              {/* Priority Badge */}
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div className={`w-10 h-10 rounded-full flex items-center justify-center font-bold text-white ${
                    rec.priority === 1 ? 'bg-red-600' :
                    rec.priority === 2 ? 'bg-orange-600' :
                    rec.priority === 3 ? 'bg-yellow-600' :
                    'bg-blue-600'
                  }`}>
                    {rec.priority}
                  </div>
                  <div>
                    <h4 className="text-lg font-semibold text-gray-900 dark:text-white">
                      {rec.action} {rec.quantity} {rec.resource_type}(s)
                    </h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      Confidence: {((rec.confidence || 0) * 100).toFixed(0)}%
                    </p>
                  </div>
                </div>
              </div>

              {/* Description */}
              <p className="text-gray-700 dark:text-gray-300 mb-4">
                {rec.description}
              </p>

              {/* Expected Impact */}
              <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4 mb-4">
                <h5 className="font-semibold text-gray-900 dark:text-white mb-2">Expected Impact</h5>
                <div className="grid grid-cols-3 gap-4 text-sm">
                  <div>
                    <div className="text-gray-600 dark:text-gray-400">DTD Reduction</div>
                    <div className="text-lg font-bold text-green-600 dark:text-green-400">
                      {rec.expected_impact.dtd_reduction}
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-600 dark:text-gray-400">LOS Reduction</div>
                    <div className="text-lg font-bold text-blue-600 dark:text-blue-400">
                      {rec.expected_impact.los_reduction}
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-600 dark:text-gray-400">LWBS Drop</div>
                    <div className="text-lg font-bold text-purple-600 dark:text-purple-400">
                      {rec.expected_impact.lwbs_drop}
                    </div>
                  </div>
                </div>
              </div>

              {/* ROI */}
              {rec.cost && rec.cost.roi_percentage > 0 && (
                <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-4 mb-4">
                  <h5 className="font-semibold text-green-900 dark:text-green-200 mb-2">💰 ROI Analysis</h5>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div>
                      <div className="text-green-700 dark:text-green-300">Cost/Shift</div>
                      <div className="text-lg font-bold text-green-900 dark:text-green-200">
                        ${rec.cost.per_shift.toFixed(0)}
                      </div>
                    </div>
                    <div>
                      <div className="text-green-700 dark:text-green-300">Cost/Year</div>
                      <div className="text-lg font-bold text-green-900 dark:text-green-200">
                        ${rec.cost.per_year.toLocaleString()}
                      </div>
                    </div>
                    <div>
                      <div className="text-green-700 dark:text-green-300">Annual Savings</div>
                      <div className="text-lg font-bold text-green-900 dark:text-green-200">
                        ${rec.cost.annual_savings.toLocaleString()}
                      </div>
                    </div>
                    <div>
                      <div className="text-green-700 dark:text-green-300">ROI</div>
                      <div className="text-lg font-bold text-green-900 dark:text-green-200">
                        {rec.cost.roi_percentage.toFixed(1)}%
                      </div>
                    </div>
                  </div>
                  <div className="mt-2 text-xs text-green-700 dark:text-green-300">
                    Payback Period: {rec.cost.payback_days.toFixed(0)} days
                  </div>
                </div>
              )}

              {/* Implementation Steps */}
              {rec.implementation_steps && rec.implementation_steps.length > 0 && (
                <div className="mb-4">
                  <h5 className="font-semibold text-gray-900 dark:text-white mb-2">📝 Implementation Steps</h5>
                  <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
                    {rec.implementation_steps.map((step, stepIdx) => (
                      <li key={stepIdx}>{step.replace(/^\d+\.\s*/, '')}</li>
                    ))}
                  </ol>
                </div>
              )}

              {/* Timeline & Risks */}
              <div className="grid md:grid-cols-2 gap-4 mb-4">
                <div>
                  <h5 className="font-semibold text-gray-900 dark:text-white mb-2">⏱️ Timeline</h5>
                  <p className="text-gray-700 dark:text-gray-300">{rec.timeline}</p>
                </div>
                {rec.risks && rec.risks.length > 0 && (
                  <div>
                    <h5 className="font-semibold text-gray-900 dark:text-white mb-2">⚠️ Risks</h5>
                    <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                      {rec.risks.map((risk, riskIdx) => (
                        <li key={riskIdx}>{risk}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>

              {/* Success Metrics */}
              {rec.success_metrics && (
                <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
                  <h5 className="font-semibold text-blue-900 dark:text-blue-200 mb-2">📊 Success Metrics</h5>
                  <div className="text-sm text-blue-800 dark:text-blue-200">
                    <p className="mb-2">
                      <strong>Primary:</strong> {rec.success_metrics.primary.metric} - Target: {rec.success_metrics.primary.target}
                    </p>
                    <p className="mb-2">
                      <strong>Success Criteria:</strong> {rec.success_metrics.success_criteria}
                    </p>
                    <p>
                      <strong>Timeline:</strong> {rec.success_metrics.timeline}
                    </p>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Next Steps */}
      {next_steps && next_steps.length > 0 && (
        <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-yellow-900 dark:text-yellow-200 mb-3">
            🚀 Next Steps
          </h3>
          <ol className="list-decimal list-inside space-y-2 text-yellow-800 dark:text-yellow-200">
            {next_steps.map((step, idx) => (
              <li key={idx} className="font-medium">{step.replace(/^\d+\.\s*/, '')}</li>
            ))}
          </ol>
        </div>
      )}

      {/* Execution Time */}
      {plan.execution_time_seconds && (
        <div className="text-xs text-gray-500 dark:text-gray-400 text-center">
          Analysis completed in {plan.execution_time_seconds.toFixed(2)} seconds
        </div>
      )}
    </div>
  )
}

