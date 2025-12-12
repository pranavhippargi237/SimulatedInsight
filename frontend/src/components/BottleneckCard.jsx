export default function BottleneckCard({ bottleneck }) {
  const severityColors = {
    low: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
    medium: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
    high: 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200',
    critical: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
  }

  const p95Wait = bottleneck.p95_wait_time_minutes
  const queueLength = bottleneck.queue_length
  const throughputDrag = bottleneck.throughput_drag_percent
  const lwbsImpact = bottleneck.lwbs_impact_percent
  const peakRange = bottleneck.peak_time_range
  const causalBreakdown = bottleneck.causal_breakdown || {}
  const equityAnalysis = bottleneck.equity_analysis || {}
  const simulatedActions = bottleneck.simulated_actions || []
  const forecast = bottleneck.forecast || {}
  const operationalExample = bottleneck.operational_example
  const flowCascade = bottleneck.flow_cascade || {}

  // Calculate impact vs benchmark (4h = 240 min)
  const benchmarkMinutes = 240
  const impactPercent = bottleneck.impact_score * 100

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 mb-6 border-l-4 border-red-500">
      {/* Header */}
      <div className="mb-4">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
            {bottleneck.bottleneck_name}
          </h2>
          <span className={`px-3 py-1 rounded-full text-sm font-bold ${severityColors[bottleneck.severity] || severityColors.medium}`}>
            {bottleneck.severity?.toUpperCase() || 'CRITICAL'}
          </span>
        </div>
        
        {/* Key Metrics Bar */}
        <div className="flex flex-wrap items-center gap-4 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-50 dark:bg-gray-700/50 p-3 rounded">
          <span>
            {bottleneck.current_wait_time_minutes?.toFixed(0) || 0} min Avg Wait
            {p95Wait && ` (P95: ${p95Wait.toFixed(0)} min)`}
          </span>
          {queueLength && <span>| Queue: {queueLength} pts</span>}
          <span>| Impact: {impactPercent.toFixed(0)}% (vs. &lt;{benchmarkMinutes/60}h benchmark)</span>
        </div>
      </div>

      {/* What's Happening */}
      <div className="mb-4">
        <h3 className="font-semibold text-lg mb-2 text-gray-900 dark:text-white">What's Happening:</h3>
        <p className="text-gray-700 dark:text-gray-300">
          Delays in {bottleneck.stage} causing {throughputDrag ? `${throughputDrag}%` : 'significant'} throughput drag
          {lwbsImpact && `—cascading to +${lwbsImpact}% LWBS (${(lwbsImpact - 0.0).toFixed(1)}% baseline spike)`}
          {peakRange && `. Peaks ${peakRange} weekends`}
          {bottleneck.metadata?.type_breakdown?.top_types?.[0] && (
            <span> ({bottleneck.metadata.type_breakdown.top_types[0][1].percentage}% {bottleneck.metadata.type_breakdown.top_types[0][0]} tests amp)</span>
          )}
        </p>
      </div>

      {/* Causal Breakdown */}
      {causalBreakdown.factors && causalBreakdown.factors.length > 0 && (
        <div className="mb-4 bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
          <h3 className="font-semibold text-lg mb-2 text-gray-900 dark:text-white">
            Causal Breakdown (SHAP Viz: Hover for paths):
          </h3>
          <div className="space-y-2">
            {causalBreakdown.factors.map((factor, idx) => (
              <div key={idx} className="text-sm text-gray-700 dark:text-gray-300">
                <strong>{factor.name}:</strong> {factor.variance_explained_percent}% variance
                {factor.ate_description && ` (${factor.ate_description}, ${factor.confidence_interval})`}
                {factor.correlation && `. ${factor.correlation}`}
                {factor.note && ` (${factor.note})`}
              </div>
            ))}
          </div>
          
          {/* Equity Slice */}
          {equityAnalysis.low_esi_impact && (
            <div className="mt-3 pt-3 border-t border-blue-200 dark:border-blue-700">
              <div className="text-sm text-gray-700 dark:text-gray-300">
                <strong>Equity Slice:</strong> {equityAnalysis.underserved_proxy_impact || equityAnalysis.low_esi_impact}
                {equityAnalysis.lwbs_risk_multiplier && ` (${equityAnalysis.lwbs_risk_multiplier} in underserved proxies)`}
              </div>
            </div>
          )}

          {/* Mini DAG */}
          {causalBreakdown.dag_paths && causalBreakdown.dag_paths.length > 0 && (
            <div className="mt-3 pt-3 border-t border-blue-200 dark:border-blue-700">
              <div className="text-xs text-gray-600 dark:text-gray-400 italic">
                [Mini DAG Embed: {causalBreakdown.dag_paths[0].path} (P={causalBreakdown.dag_paths[0].probability} via {causalBreakdown.dag_paths[0].method})]
              </div>
            </div>
          )}
        </div>
      )}

      {/* Simulated Actions Table */}
      {simulatedActions.length > 0 && (
        <div className="mb-4">
          <h3 className="font-semibold text-lg mb-2 text-gray-900 dark:text-white">
            Simmed Actions (Top {simulatedActions.length} Prioritized; Run What-If?):
          </h3>
          <div className="overflow-x-auto">
            <table className="min-w-full border-collapse border border-gray-300 dark:border-gray-600">
              <thead>
                <tr className="bg-gray-100 dark:bg-gray-700">
                  <th className="border border-gray-300 dark:border-gray-600 px-3 py-2 text-left text-sm font-semibold">Action</th>
                  <th className="border border-gray-300 dark:border-gray-600 px-3 py-2 text-left text-sm font-semibold">Delta</th>
                  <th className="border border-gray-300 dark:border-gray-600 px-3 py-2 text-left text-sm font-semibold">ROI</th>
                  <th className="border border-gray-300 dark:border-gray-600 px-3 py-2 text-left text-sm font-semibold">Conf</th>
                  <th className="border border-gray-300 dark:border-gray-600 px-3 py-2 text-left text-sm font-semibold">Equity Lift</th>
                </tr>
              </thead>
              <tbody>
                {simulatedActions.map((action, idx) => (
                  <tr key={idx} className="hover:bg-gray-50 dark:hover:bg-gray-700/50">
                    <td className="border border-gray-300 dark:border-gray-600 px-3 py-2 text-sm">
                      {action.priority}. {action.action}
                    </td>
                    <td className="border border-gray-300 dark:border-gray-600 px-3 py-2 text-sm">
                      {action.delta_description || `${action.delta_wait_minutes} min wait`}
                    </td>
                    <td className="border border-gray-300 dark:border-gray-600 px-3 py-2 text-sm">
                      {action.roi_description || `$${action.roi_per_day/1000}k/day`}
                    </td>
                    <td className="border border-gray-300 dark:border-gray-600 px-3 py-2 text-sm">
                      {action.confidence_percent}%
                    </td>
                    <td className="border border-gray-300 dark:border-gray-600 px-3 py-2 text-sm">
                      {action.equity_lift || 'Neutral'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Forecast */}
      {forecast.description && (
        <div className="mb-4 bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg">
          <h3 className="font-semibold text-lg mb-2 text-gray-900 dark:text-white">Forecast:</h3>
          <p className="text-gray-700 dark:text-gray-300">{forecast.description}</p>
        </div>
      )}

      {/* Operational Example */}
      {operationalExample && (
        <div className="mb-4 bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
          <h3 className="font-semibold text-lg mb-2 text-gray-900 dark:text-white">Operational Example:</h3>
          <p className="text-gray-700 dark:text-gray-300">{operationalExample}</p>
        </div>
      )}

      {/* Patient Flow Cascade */}
      {flowCascade.cascade_paths && flowCascade.cascade_paths.length > 0 && (
        <div className="mb-4 bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg border border-purple-200 dark:border-purple-800">
          <h3 className="font-semibold text-lg mb-3 text-gray-900 dark:text-white">
            Patient Flow Cascade:
          </h3>
          
          {/* Cascade Paths */}
          <div className="space-y-3 mb-4">
            {flowCascade.cascade_paths.map((path, idx) => (
              <div key={idx} className="bg-white dark:bg-gray-800 p-3 rounded border border-purple-200 dark:border-purple-700">
                <div className="flex items-center gap-2 mb-2">
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
                {flowCascade.downstream_impacts?.[path.to] && (
                  <div className="text-xs text-gray-600 dark:text-gray-400 ml-6 mt-1">
                    {flowCascade.downstream_impacts[path.to].impact_description}
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* System Impact Summary */}
          {flowCascade.system_impact && (
            <div className="mt-4 pt-4 border-t border-purple-200 dark:border-purple-700">
              <div className="text-sm text-gray-700 dark:text-gray-300">
                <strong>System Impact:</strong> {flowCascade.system_impact.total_patients_affected} patients affected, 
                {flowCascade.system_impact.total_delay_minutes} min total delay, 
                {flowCascade.system_impact.cascade_depth} cascade paths ({flowCascade.system_impact.severity} severity)
              </div>
            </div>
          )}

          {/* Upstream Dependencies */}
          {flowCascade.upstream_dependencies && flowCascade.upstream_dependencies.length > 0 && (
            <div className="mt-4 pt-4 border-t border-purple-200 dark:border-purple-700">
              <div className="text-sm font-medium text-gray-900 dark:text-white mb-2">Upstream Dependencies:</div>
              <div className="space-y-1">
                {flowCascade.upstream_dependencies.map((dep, idx) => (
                  <div key={idx} className="text-xs text-gray-600 dark:text-gray-400">
                    • {dep.description} (strength: {dep.dependency_strength})
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Root Causes */}
      {bottleneck.causes && bottleneck.causes.length > 0 && (
        <div className="mb-4">
          <h3 className="font-semibold text-lg mb-2 text-gray-900 dark:text-white">Root causes:</h3>
          <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
            {bottleneck.causes.map((cause, idx) => (
              <li key={idx}>{cause}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Recommended Actions */}
      {bottleneck.recommendations && bottleneck.recommendations.length > 0 && (
        <div className="mb-4 bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
          <h3 className="font-semibold text-lg mb-2 text-gray-900 dark:text-white">Recommended Actions:</h3>
          <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
            {bottleneck.recommendations.map((rec, idx) => (
              <li key={idx}>{rec}</li>
            ))}
          </ol>
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex gap-2 mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
        <button className="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded">
          📥 Export PDF/CSV
        </button>
        <button className="px-3 py-1 bg-green-600 hover:bg-green-700 text-white text-sm rounded">
          🔄 Rerun Sim
        </button>
        <button className="px-3 py-1 bg-purple-600 hover:bg-purple-700 text-white text-sm rounded">
          ❓ NL Drill: "Equity details?"
        </button>
      </div>
    </div>
  )
}
