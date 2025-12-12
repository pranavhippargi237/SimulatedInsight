export default function SimulationResult({ result }) {
  if (!result) return null

  const baseline = result.baseline_metrics || {}
  const predicted = result.predicted_metrics || {}
  const deltas = result.deltas || {}

  // Calculate actual changes
  const dtdChange = deltas.dtd_reduction || deltas.dtd_change || 0
  const losChange = deltas.los_reduction || deltas.los_change || 0
  const lwbsChange = deltas.lwbs_drop || deltas.lwbs_change || 0

  const formatChange = (value) => {
    // For dtd_reduction and los_reduction: positive = improvement (reduction in time)
    // For lwbs_drop: positive = improvement (reduction in rate)
    const absValue = Math.abs(value)
    if (value > 0) {
      return `reduced by ${absValue.toFixed(1)}%`
    } else if (value < 0) {
      return `increased by ${absValue.toFixed(1)}%`
    }
    return 'unchanged'
  }

  const getImpactLevel = (change) => {
    const absChange = Math.abs(change)
    if (absChange > 20) return { level: 'significant', color: 'text-green-600 dark:text-green-400', icon: '🎯' }
    if (absChange > 10) return { level: 'moderate', color: 'text-blue-600 dark:text-blue-400', icon: '✓' }
    if (absChange > 5) return { level: 'small', color: 'text-yellow-600 dark:text-yellow-400', icon: '→' }
    return { level: 'minimal', color: 'text-gray-600 dark:text-gray-400', icon: '○' }
  }

  const dtdImpact = getImpactLevel(dtdChange)
  const losImpact = getImpactLevel(losChange)
  const lwbsImpact = getImpactLevel(lwbsChange)

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
      <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">Simulation Results</h3>
      
      {/* Summary */}
      <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4 mb-6">
        <h4 className="font-semibold text-blue-900 dark:text-blue-200 mb-2">Summary</h4>
        <p className="text-blue-800 dark:text-blue-200">
          Based on the simulation scenario, here's what would happen to your ED operations:
        </p>
      </div>

      {/* Door-to-Doctor Time */}
      <div className="mb-6 p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
        <div className="flex items-start justify-between mb-2">
          <div>
            <h4 className="text-lg font-semibold text-gray-900 dark:text-white mb-1">
              Door-to-Doctor Time (DTD)
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Time from patient arrival to first doctor contact
            </p>
          </div>
          <div className="text-right">
            <div className={`text-2xl font-bold ${dtdImpact.color}`}>
              {dtdImpact.icon} {dtdChange > 0 ? '↓' : dtdChange < 0 ? '↑' : '→'}
            </div>
          </div>
        </div>
        <div className="mt-3 space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-gray-600 dark:text-gray-400">Current (Baseline):</span>
            <span className="font-medium text-gray-900 dark:text-white">{baseline.dtd?.toFixed(1) || 0} minutes</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-gray-600 dark:text-gray-400">Predicted:</span>
            <span className="font-medium text-gray-900 dark:text-white">{predicted.dtd?.toFixed(1) || 0} minutes</span>
          </div>
          <div className="pt-2 border-t border-gray-200 dark:border-gray-600">
            <p className="text-gray-700 dark:text-gray-300">
              <span className="font-medium">Impact: </span>
              DTD would be <span className={`font-semibold ${dtdImpact.color}`}>{formatChange(dtdChange)}</span>.
              {dtdChange > 10 && (
                <span className="block mt-1 text-green-600 dark:text-green-400">
                  ✓ This is a significant improvement that would reduce patient wait times substantially.
                </span>
              )}
              {dtdChange < -10 && (
                <span className="block mt-1 text-red-600 dark:text-red-400">
                  ⚠️ This would increase wait times, potentially worsening patient satisfaction.
                </span>
              )}
              {Math.abs(dtdChange) < 5 && (
                <span className="block mt-1 text-gray-600 dark:text-gray-400">
                  → Minimal impact. Consider other bottlenecks or resource constraints.
                </span>
              )}
            </p>
            <p className="text-xs text-gray-500 dark:text-gray-500 mt-2">
              <span className="font-medium">Reasoning: </span>
              {dtdChange > 0 
                ? `Adding ${result.parsed_scenario?.quantity || 1} ${result.parsed_scenario?.resource_type || 'resource'}(s) reduces wait times by decreasing queue length and allowing faster patient processing.`
                : dtdChange < 0
                ? `This change may increase wait times due to resource constraints or system bottlenecks elsewhere. Consider checking triage capacity, lab/imaging availability, or bed availability.`
                : `This change has minimal impact. The bottleneck may be elsewhere (e.g., triage, labs, imaging, or beds) rather than doctor availability.`
              }
            </p>
          </div>
        </div>
      </div>

      {/* Length of Stay */}
      <div className="mb-6 p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
        <div className="flex items-start justify-between mb-2">
          <div>
            <h4 className="text-lg font-semibold text-gray-900 dark:text-white mb-1">
              Length of Stay (LOS)
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Total time patients spend in the ED
            </p>
          </div>
          <div className="text-right">
            <div className={`text-2xl font-bold ${losImpact.color}`}>
              {losImpact.icon} {losChange > 0 ? '↓' : losChange < 0 ? '↑' : '→'}
            </div>
          </div>
        </div>
        <div className="mt-3 space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-gray-600 dark:text-gray-400">Current (Baseline):</span>
            <span className="font-medium text-gray-900 dark:text-white">{baseline.los?.toFixed(1) || 0} minutes</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-gray-600 dark:text-gray-400">Predicted:</span>
            <span className="font-medium text-gray-900 dark:text-white">{predicted.los?.toFixed(1) || 0} minutes</span>
          </div>
          <div className="pt-2 border-t border-gray-200 dark:border-gray-600">
            <p className="text-gray-700 dark:text-gray-300">
              <span className="font-medium">Impact: </span>
              LOS would be <span className={`font-semibold ${losImpact.color}`}>{formatChange(losChange)}</span>.
              {losChange > 10 && (
                <span className="block mt-1 text-green-600 dark:text-green-400">
                  ✓ This would improve throughput and reduce crowding in the ED.
                </span>
              )}
              {losChange < -10 && (
                <span className="block mt-1 text-red-600 dark:text-red-400">
                  ⚠️ This would increase ED occupancy, potentially causing boarding issues.
                </span>
              )}
            </p>
            <p className="text-xs text-gray-500 dark:text-gray-500 mt-2">
              <span className="font-medium">Why this happens: </span>
              {losChange > 0
                ? `Adding ${result.parsed_scenario?.quantity || 1} ${result.parsed_scenario?.resource_type || 'resource'}(s) reduces LOS by (1) faster doctor visits (reduced service time), (2) faster DTD (patients move through system quicker), and (3) reduced downstream bottlenecks. However, LOS is also affected by labs, imaging, and bed availability - if those are constrained, LOS improvement may be limited.`
                : losChange < 0
                ? `This change increases LOS. Possible reasons: (1) Downstream bottlenecks (labs, imaging, beds) are the constraint, not doctor availability, (2) Adding doctors may shift the bottleneck elsewhere, or (3) System flow dynamics create unexpected delays. Check lab/imaging capacity and bed availability.`
                : `This change has minimal impact on LOS. LOS is primarily driven by downstream processes (labs, imaging, bed availability) rather than doctor availability. The bottleneck is likely in those areas.`
              }
            </p>
          </div>
        </div>
      </div>

      {/* LWBS Rate */}
      <div className="mb-6 p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
        <div className="flex items-start justify-between mb-2">
          <div>
            <h4 className="text-lg font-semibold text-gray-900 dark:text-white mb-1">
              Left Without Being Seen (LWBS) Rate
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Percentage of patients who leave before seeing a doctor
            </p>
          </div>
          <div className="text-right">
            <div className={`text-2xl font-bold ${lwbsImpact.color}`}>
              {lwbsImpact.icon} {lwbsChange > 0 ? '↓' : lwbsChange < 0 ? '↑' : '→'}
            </div>
          </div>
        </div>
        <div className="mt-3 space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-gray-600 dark:text-gray-400">Current (Baseline):</span>
            <span className="font-medium text-gray-900 dark:text-white">{((baseline.lwbs || 0) * 100).toFixed(1)}%</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-gray-600 dark:text-gray-400">Predicted:</span>
            <span className="font-medium text-gray-900 dark:text-white">{((predicted.lwbs || 0) * 100).toFixed(1)}%</span>
          </div>
          <div className="pt-2 border-t border-gray-200 dark:border-gray-600">
            <p className="text-gray-700 dark:text-gray-300">
              <span className="font-medium">Impact: </span>
              LWBS rate would be <span className={`font-semibold ${lwbsImpact.color}`}>{formatChange(lwbsChange)}</span>.
              {lwbsChange > 20 && (
                <span className="block mt-1 text-green-600 dark:text-green-400">
                  ✓ This is a major improvement that would significantly reduce patient walkouts.
                </span>
              )}
              {lwbsChange < -20 && (
                <span className="block mt-1 text-red-600 dark:text-red-400">
                  ⚠️ This would increase patient walkouts, indicating serious capacity issues.
                </span>
              )}
            </p>
            <p className="text-xs text-gray-500 dark:text-gray-500 mt-2">
              <span className="font-medium">Why this happens: </span>
              {lwbsChange > 0
                ? `Adding ${result.parsed_scenario?.quantity || 1} ${result.parsed_scenario?.resource_type || 'resource'}(s) reduces LWBS by decreasing DTD (door-to-doctor time). Patients who wait less than their ESI-specific threshold (ESI-3: 45 min, ESI-4: 30 min, ESI-5: 20 min) are less likely to leave. However, if DTD reduction is minimal, LWBS improvement will also be minimal.`
                : lwbsChange < 0
                ? `This change increases LWBS risk. This is unusual and suggests: (1) DTD actually increased (check DTD metric above), (2) The bottleneck shifted to triage (patients wait longer before even queuing for doctor), or (3) System flow dynamics created unexpected delays.`
                : `This change has minimal impact on LWBS. LWBS is primarily driven by DTD - if DTD reduction is small, LWBS improvement will be minimal. Also, LWBS only affects low-acuity patients (ESI 3-5) who wait beyond their threshold.`
              }
            </p>
          </div>
        </div>
      </div>

      {/* Overall Assessment */}
      <div className="bg-gradient-to-r from-blue-50 to-green-50 dark:from-blue-900/20 dark:to-green-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4 mb-4">
        <h4 className="font-semibold text-gray-900 dark:text-white mb-2">Overall Assessment</h4>
        <p className="text-gray-700 dark:text-gray-300 mb-2">
          {(dtdChange > 10 || losChange > 10 || lwbsChange > 5) && (dtdChange > 0 && losChange > 0 && lwbsChange >= 0) ? (
            <span>
              <span className="font-bold text-green-600 dark:text-green-400">✓ STRONGLY RECOMMENDED: </span>
              This scenario would significantly improve ED operations. You should expect:
              <ul className="list-disc list-inside mt-2 space-y-1">
                {dtdChange > 10 && <li>Faster patient care (DTD reduced by {dtdChange.toFixed(1)}%)</li>}
                {losChange > 10 && <li>Better throughput (LOS reduced by {losChange.toFixed(1)}%)</li>}
                {lwbsChange > 5 && <li>Fewer walkouts (LWBS reduced by {lwbsChange.toFixed(1)}%)</li>}
              </ul>
            </span>
          ) : (dtdChange > 5 || losChange > 5 || lwbsChange > 2) && (dtdChange > 0 || losChange > 0 || lwbsChange > 0) ? (
            <span>
              <span className="font-bold text-blue-600 dark:text-blue-400">✓ RECOMMENDED: </span>
              This scenario would provide measurable improvements to ED operations. Key benefits: {dtdChange > 5 ? `DTD reduced by ${dtdChange.toFixed(1)}%` : ''}{dtdChange > 5 && losChange > 5 ? ', ' : ''}{losChange > 5 ? `LOS reduced by ${losChange.toFixed(1)}%` : ''}{(dtdChange > 5 || losChange > 5) && lwbsChange > 2 ? ', ' : ''}{lwbsChange > 2 ? `LWBS reduced by ${lwbsChange.toFixed(1)}%` : ''}.
            </span>
          ) : (dtdChange < -10 || losChange < -10 || lwbsChange < -5) ? (
            <span>
              <span className="font-bold text-red-600 dark:text-red-400">⚠️ NOT RECOMMENDED: </span>
              This scenario would worsen ED operations. Key concerns: {dtdChange < -10 ? `DTD increased by ${Math.abs(dtdChange).toFixed(1)}%` : ''}{dtdChange < -10 && losChange < -10 ? ', ' : ''}{losChange < -10 ? `LOS increased by ${Math.abs(losChange).toFixed(1)}%` : ''}{(dtdChange < -10 || losChange < -10) && lwbsChange < -5 ? ', ' : ''}{lwbsChange < -5 ? `LWBS increased by ${Math.abs(lwbsChange).toFixed(1)}%` : ''}. Consider alternative approaches or addressing downstream bottlenecks first.
            </span>
          ) : (
            <span>
              <span className="font-bold text-gray-600 dark:text-gray-400">→ MINIMAL IMPACT: </span>
              This scenario would have limited impact on ED operations. Review specific metrics above to understand trade-offs. Consider whether the resource investment is justified given the limited improvement, or if bottlenecks are elsewhere (triage, labs, imaging, beds).
            </span>
          )}
        </p>
      </div>

      {/* Technical Details */}
      <div className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
        <p>
          <span className="font-medium">Confidence: </span>
          {((result?.confidence ?? 0) * 100).toFixed(1)}% (based on simulation variance)
        </p>
        <p>
          <span className="font-medium">Execution Time: </span>
          {Number(result?.execution_time_seconds ?? 0).toFixed(2)} seconds
        </p>
        {result.scenario_id && (
          <p>
            <span className="font-medium">Simulation ID: </span>
            {result.scenario_id}
          </p>
        )}
      </div>
    </div>
  )
}
