import React, { useState } from 'react'

// Heroicons - fallback to simple icons if not available
const ChevronDownIcon = ({ className }) => <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" /></svg>
const ChevronRightIcon = ({ className }) => <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" /></svg>
const ChartBarIcon = ({ className }) => <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" /></svg>
const LightBulbIcon = ({ className }) => <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" /></svg>
const ExclamationTriangleIcon = ({ className }) => <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>
const CurrencyDollarIcon = ({ className }) => <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
import CausalDAG from './CausalDAG'
import SHAPHeatmap from './SHAPHeatmap'
import PatientFlowSankey from './PatientFlowSankey'
import { getPatientFlow } from '../services/api'

export default function StructuredAnalysis({ analysis, bottleneck }) {
  const [expandedSections, setExpandedSections] = useState(new Set(['executive_summary']))
  const [selectedInsight, setSelectedInsight] = useState(null)

  const toggleSection = (sectionId) => {
    const newExpanded = new Set(expandedSections)
    if (newExpanded.has(sectionId)) {
      newExpanded.delete(sectionId)
    } else {
      newExpanded.add(sectionId)
    }
    setExpandedSections(newExpanded)
  }

  if (!analysis) {
    return (
      <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-8 text-center">
        <p className="text-gray-500 dark:text-gray-400">No analysis data available</p>
      </div>
    )
  }

  const sections = [
    {
      id: 'executive_summary',
      title: 'Executive Summary',
      icon: ChartBarIcon,
      color: 'blue',
      content: (
        <div className="space-y-4">
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
              <div className="text-sm text-blue-600 dark:text-blue-400">Current Value</div>
              <div className="text-2xl font-bold text-blue-900 dark:text-blue-100">
                {analysis.current_value?.toFixed(1) || 'N/A'}
              </div>
              <div className="text-xs text-blue-700 dark:text-blue-300 mt-1">
                {analysis.metric_name?.toUpperCase() || 'METRIC'}
              </div>
            </div>
            <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
              <div className="text-sm text-green-600 dark:text-green-400">Benchmark</div>
              <div className="text-2xl font-bold text-green-900 dark:text-green-100">
                {analysis.benchmark_value?.toFixed(1) || 'N/A'}
              </div>
              <div className="text-xs text-green-700 dark:text-green-300 mt-1">
                2025 Target
              </div>
            </div>
            <div className={`p-4 rounded-lg ${
              (analysis.current_value || 0) > (analysis.benchmark_value || 0)
                ? 'bg-red-50 dark:bg-red-900/20'
                : 'bg-green-50 dark:bg-green-900/20'
            }`}>
              <div className={`text-sm ${
                (analysis.current_value || 0) > (analysis.benchmark_value || 0)
                  ? 'text-red-600 dark:text-red-400'
                  : 'text-green-600 dark:text-green-400'
              }`}>
                Variance
              </div>
              <div className={`text-2xl font-bold ${
                (analysis.current_value || 0) > (analysis.benchmark_value || 0)
                  ? 'text-red-900 dark:text-red-100'
                  : 'text-green-900 dark:text-green-100'
              }`}>
                {((analysis.current_value || 0) - (analysis.benchmark_value || 0)).toFixed(1)}
              </div>
              <div className={`text-xs mt-1 ${
                (analysis.current_value || 0) > (analysis.benchmark_value || 0)
                  ? 'text-red-700 dark:text-red-300'
                  : 'text-green-700 dark:text-green-300'
              }`}>
                {((analysis.current_value || 0) > (analysis.benchmark_value || 0)) ? 'Above Target' : 'Below Target'}
              </div>
            </div>
          </div>
          
          {analysis.excess !== undefined && (
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4">
              <div className="text-sm font-semibold text-yellow-900 dark:text-yellow-200 mb-2">
                Key Finding
              </div>
              <div className="text-sm text-yellow-800 dark:text-yellow-300">
                Current performance is {Math.abs(analysis.excess).toFixed(1)} {analysis.metric_name === 'lwbs' ? 'percentage points' : 'minutes'} 
                {analysis.excess > 0 ? ' above' : ' below'} the 2025 benchmark target.
              </div>
            </div>
          )}
        </div>
      )
    },
    {
      id: 'key_insights',
      title: 'Key Insights',
      icon: LightBulbIcon,
      color: 'purple',
      count: analysis.insights?.length || 0,
            content: (
        <div className="space-y-3">
          {analysis.insights && Array.isArray(analysis.insights) && analysis.insights.length > 0 ? (
            analysis.insights.map((insight, idx) => (
              <div
                key={idx}
                className={`border rounded-lg p-4 cursor-pointer transition-all hover:shadow-md ${
                  selectedInsight === idx
                    ? 'border-purple-500 bg-purple-50 dark:bg-purple-900/20'
                    : 'border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800'
                }`}
                onClick={() => setSelectedInsight(selectedInsight === idx ? null : idx)}
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-xs font-semibold text-purple-600 dark:text-purple-400 bg-purple-100 dark:bg-purple-900/30 px-2 py-1 rounded">
                        {insight.insight_type || insight.type || 'INSIGHT'}
                      </span>
                      <span className="text-xs text-gray-500 dark:text-gray-400">
                        {((insight.impact_score || 0) * 100).toFixed(0)}% Impact
                      </span>
                      <span className="text-xs text-gray-500 dark:text-gray-400">
                        {((insight.confidence || 0) * 100).toFixed(0)}% Confidence
                      </span>
                    </div>
                    <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-1">
                      {insight.title}
                    </h4>
                    <p className="text-sm text-gray-700 dark:text-gray-300">
                      {insight.description}
                    </p>
                  </div>
                </div>
                
                {selectedInsight === idx && (
                  <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700 space-y-3">
                    {insight.evidence && Object.keys(insight.evidence).length > 0 && (
                      <div>
                        <div className="text-xs font-semibold text-gray-600 dark:text-gray-400 mb-2">
                          Evidence
                        </div>
                        <div className="bg-gray-50 dark:bg-gray-900 rounded p-3 text-xs">
                          <pre className="whitespace-pre-wrap text-gray-700 dark:text-gray-300">
                            {JSON.stringify(insight.evidence, null, 2)}
                          </pre>
                        </div>
                      </div>
                    )}
                    
                    {insight.recommendation && (
                      <div>
                        <div className="text-xs font-semibold text-green-600 dark:text-green-400 mb-2">
                          Recommendation
                        </div>
                        <div className="bg-green-50 dark:bg-green-900/20 rounded p-3 text-sm text-green-800 dark:text-green-300">
                          {insight.recommendation}
                        </div>
                      </div>
                    )}
                    
                    {insight.unmet_need && (
                      <div>
                        <div className="text-xs font-semibold text-orange-600 dark:text-orange-400 mb-2">
                          Unmet Need
                        </div>
                        <div className="bg-orange-50 dark:bg-orange-900/20 rounded p-3 text-sm text-orange-800 dark:text-orange-300">
                          {insight.unmet_need}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))
          ) : (
            <div className="text-center py-8 space-y-2">
              <p className="text-gray-500 dark:text-gray-400 font-medium">
                No insights available
              </p>
              <p className="text-xs text-gray-400 dark:text-gray-500">
                {analysis.metric_name === "ED Operations" || !analysis.metric_name
                  ? "Try asking about specific metrics like 'What are my bottlenecks?' or 'Analyze LWBS'"
                  : "Insufficient data or no patterns detected. Upload more data or try a different time window."}
              </p>
            </div>
          )}
        </div>
      )
    },
    {
      id: 'root_causes',
      title: 'Root Cause Analysis',
      icon: ExclamationTriangleIcon,
      color: 'red',
      count: analysis.root_causes?.length || 0,
      content: (
        <div className="space-y-3">
          {analysis.root_causes && analysis.root_causes.length > 0 ? (
            analysis.root_causes.map((cause, idx) => (
              <div
                key={idx}
                className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4"
              >
                <div className="flex items-start gap-3">
                  <div className="flex-shrink-0 w-6 h-6 bg-red-500 text-white rounded-full flex items-center justify-center text-xs font-bold">
                    {idx + 1}
                  </div>
                  <div className="flex-1">
                    <div className="text-sm font-semibold text-red-900 dark:text-red-200 mb-1">
                      {typeof cause === 'string' ? cause : cause.title || cause.description}
                    </div>
                    {typeof cause === 'object' && cause.description && (
                      <div className="text-xs text-red-800 dark:text-red-300">
                        {cause.description}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))
          ) : (
            <div className="text-center py-8 space-y-2">
              <p className="text-gray-500 dark:text-gray-400 font-medium">
                No root causes identified
              </p>
              <p className="text-xs text-gray-400 dark:text-gray-500">
                {analysis.insights && analysis.insights.length > 0
                  ? "Root cause analysis may require more detailed data"
                  : "Upload data and ask specific questions to identify root causes"}
              </p>
            </div>
          )}
        </div>
      )
    },
    {
      id: 'patterns',
      title: 'Patterns & Trends',
      icon: ChartBarIcon,
      color: 'indigo',
      content: (
        <div className="space-y-4">
          {analysis.patterns && Object.keys(analysis.patterns).length > 0 ? (
            Object.entries(analysis.patterns).map(([key, value]) => (
              <div key={key} className="bg-indigo-50 dark:bg-indigo-900/20 border border-indigo-200 dark:border-indigo-800 rounded-lg p-4">
                <div className="text-sm font-semibold text-indigo-900 dark:text-indigo-200 mb-2">
                  {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </div>
                <div className="text-sm text-indigo-800 dark:text-indigo-300">
                  {typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value)}
                </div>
              </div>
            ))
          ) : (
            <div className="text-center py-8 space-y-2">
              <p className="text-gray-500 dark:text-gray-400 font-medium">
                No patterns detected
              </p>
              <p className="text-xs text-gray-400 dark:text-gray-500">
                Patterns require sufficient data points. Try analyzing a longer time window or upload more data.
              </p>
            </div>
          )}
        </div>
      )
    },
    {
      id: 'causal_analysis',
      title: 'Causal Analysis',
      icon: ChartBarIcon,
      color: 'purple',
      content: bottleneck?.metadata?.causal_analysis ? (
        <div className="space-y-4">
          <CausalDAG 
            causalData={bottleneck.metadata.causal_analysis} 
            bottleneckName={bottleneck.bottleneck_name}
          />
          <SHAPHeatmap 
            shapData={bottleneck.metadata.causal_analysis.feature_attributions}
            featureNames={Object.keys(bottleneck.metadata.causal_analysis.feature_attributions?.attributions || {})}
          />
        </div>
      ) : (
        <div className="text-center py-8 text-gray-500 dark:text-gray-400">
          No causal analysis available
        </div>
      )
    },
    {
      id: 'economic_impact',
      title: 'Economic Impact',
      icon: CurrencyDollarIcon,
      color: 'green',
      content: analysis.economic_impact ? (
        <div className="space-y-4">
          {Object.entries(analysis.economic_impact).map(([key, value]) => (
            <div key={key} className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-4">
              <div className="text-sm font-semibold text-green-900 dark:text-green-200 mb-1">
                {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
              </div>
              <div className="text-lg font-bold text-green-800 dark:text-green-300">
                {typeof value === 'number' 
                  ? (key.includes('cost') || key.includes('savings') || key.includes('revenue') 
                      ? `$${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
                      : value.toFixed(2))
                  : String(value)}
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="text-center py-8 text-gray-500 dark:text-gray-400">
          No economic impact data available
        </div>
      )
    },
    {
      id: 'predictive_signals',
      title: 'Predictive Signals',
      icon: ChartBarIcon,
      color: 'amber',
      content: analysis.predictive_signals && Object.keys(analysis.predictive_signals).length > 0 ? (
        <div className="space-y-4">
          {Object.entries(analysis.predictive_signals).map(([key, value]) => (
            <div key={key} className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg p-4">
              <div className="text-sm font-semibold text-amber-900 dark:text-amber-200 mb-2">
                {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
              </div>
              <div className="text-sm text-amber-800 dark:text-amber-300">
                {typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value)}
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="text-center py-8 text-gray-500 dark:text-gray-400">
          No predictive signals detected
        </div>
      )
    }
  ]

  return (
    <div className="space-y-4">
      {sections.map((section) => {
        const Icon = section.icon
        const isExpanded = expandedSections.has(section.id)
        const colorClasses = {
          blue: 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800 text-blue-900 dark:text-blue-200',
          purple: 'bg-purple-50 dark:bg-purple-900/20 border-purple-200 dark:border-purple-800 text-purple-900 dark:text-purple-200',
          red: 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800 text-red-900 dark:text-red-200',
          indigo: 'bg-indigo-50 dark:bg-indigo-900/20 border-indigo-200 dark:border-indigo-800 text-indigo-900 dark:text-indigo-200',
          green: 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800 text-green-900 dark:text-green-200',
          amber: 'bg-amber-50 dark:bg-amber-900/20 border-amber-200 dark:border-amber-800 text-amber-900 dark:text-amber-200'
        }

        return (
          <div
            key={section.id}
            className={`border rounded-lg overflow-hidden ${colorClasses[section.color]}`}
          >
            <button
              onClick={() => toggleSection(section.id)}
              className="w-full flex items-center justify-between p-4 hover:bg-opacity-80 transition-colors"
            >
              <div className="flex items-center gap-3">
                <Icon className="w-5 h-5" />
                <span className="font-semibold">{section.title}</span>
                {section.count !== undefined && (
                  <span className="text-xs bg-white dark:bg-gray-800 px-2 py-1 rounded">
                    {section.count}
                  </span>
                )}
              </div>
              {isExpanded ? (
                <ChevronDownIcon className="w-5 h-5" />
              ) : (
                <ChevronRightIcon className="w-5 h-5" />
              )}
            </button>
            
            {isExpanded && (
              <div className="px-4 pb-4 border-t border-current border-opacity-20">
                {section.content}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
