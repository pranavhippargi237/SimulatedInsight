import React, { useCallback } from 'react'
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  addEdge,
  MarkerType
} from 'reactflow'
import 'reactflow/dist/style.css'
import SHAPHeatmap from './SHAPHeatmap'

export default function CausalDAG({ causalData, bottleneckName }) {
  // Parse DAG structure from causal analysis
  const parseDAG = useCallback(() => {
    if (!causalData) return { nodes: [], edges: [] }
    
    const nodes = []
    const edges = []
    const nodePositions = {}
    let yPos = 0
    
    // Extract nodes from causal graph or feature attributions
    const attributions = causalData.feature_attributions?.attributions || {}
    const ate = causalData.ate_estimates || {}
    const counterfactuals = causalData.counterfactuals || []
    
    // Create nodes from attributions
    const sortedAttribs = Object.entries(attributions)
      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
      .slice(0, 8) // Top 8 factors
    
    sortedAttribs.forEach(([factor, value], idx) => {
      const nodeId = `node-${idx}`
      const isPositive = value > 0
      
      nodes.push({
        id: nodeId,
        type: 'default',
        data: {
          label: (
            <div className="text-xs">
              <div className="font-semibold">{factor.replace(/_/g, ' ')}</div>
              <div className={`text-xs ${isPositive ? 'text-red-600' : 'text-green-600'}`}>
                {value > 0 ? '+' : ''}{value.toFixed(1)}%
              </div>
            </div>
          ),
        },
        position: { x: (idx % 4) * 200, y: Math.floor(idx / 4) * 120 },
        style: {
          background: isPositive ? '#fee2e2' : '#dcfce7',
          border: `2px solid ${isPositive ? '#ef4444' : '#22c55e'}`,
          borderRadius: '8px',
          padding: '10px',
          minWidth: '150px',
        },
      })
      
      nodePositions[factor] = nodeId
    })
    
    // Add outcome node (bottleneck)
    const outcomeNode = {
      id: 'outcome',
      type: 'default',
      data: {
        label: (
          <div className="text-sm font-bold">
            <div>{bottleneckName || 'Bottleneck'}</div>
            <div className="text-xs text-gray-600">Wait Time</div>
          </div>
        ),
      },
      position: { x: 400, y: Math.max(240, sortedAttribs.length * 30) },
      style: {
        background: '#dbeafe',
        border: '3px solid #3b82f6',
        borderRadius: '12px',
        padding: '15px',
        fontWeight: 'bold',
        minWidth: '180px',
      },
    }
    nodes.push(outcomeNode)
    
    // Create edges from factors to outcome
    sortedAttribs.forEach(([factor], idx) => {
      const nodeId = `node-${idx}`
      const value = attributions[factor]
      const isPositive = value > 0
      
      edges.push({
        id: `edge-${idx}`,
        source: nodeId,
        target: 'outcome',
        type: 'smoothstep',
        animated: true,
        style: {
          stroke: isPositive ? '#ef4444' : '#22c55e',
          strokeWidth: Math.abs(value) / 10 + 1,
        },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: isPositive ? '#ef4444' : '#22c55e',
        },
        label: `${value > 0 ? '+' : ''}${value.toFixed(1)}%`,
        labelStyle: {
          fill: isPositive ? '#ef4444' : '#22c55e',
          fontWeight: 600,
          fontSize: '10px',
        },
      })
    })
    
    // Add counterfactual insights as annotation nodes
    if (counterfactuals.length > 0) {
      const topCounterfactual = counterfactuals[0]
      const cfNode = {
        id: 'counterfactual',
        type: 'default',
        data: {
          label: (
            <div className="text-xs">
              <div className="font-semibold text-purple-700">Counterfactual</div>
              <div className="text-purple-600">
                {topCounterfactual.scenario || 'Intervention'}
              </div>
              <div className="text-xs text-gray-600 mt-1">
                Expected: {topCounterfactual.expected_outcome || 'N/A'}
              </div>
            </div>
          ),
        },
        position: { x: 650, y: Math.max(240, sortedAttribs.length * 30) },
        style: {
          background: '#f3e8ff',
          border: '2px dashed #a855f7',
          borderRadius: '8px',
          padding: '10px',
          minWidth: '150px',
        },
      }
      nodes.push(cfNode)
      
      edges.push({
        id: 'edge-cf',
        source: 'outcome',
        target: 'counterfactual',
        type: 'smoothstep',
        style: {
          stroke: '#a855f7',
          strokeWidth: 2,
          strokeDasharray: '5,5',
        },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: '#a855f7',
        },
      })
    }
    
    return { nodes, edges }
  }, [causalData, bottleneckName])
  
  const [nodes, setNodes, onNodesChange] = useNodesState([])
  const [edges, setEdges, onEdgesChange] = useEdgesState([])
  
  React.useEffect(() => {
    const { nodes: parsedNodes, edges: parsedEdges } = parseDAG()
    setNodes(parsedNodes)
    setEdges(parsedEdges)
  }, [causalData, bottleneckName, parseDAG, setNodes, setEdges])
  
  const onConnect = useCallback(
    (params) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  )
  
  if (!causalData || nodes.length === 0) {
    return (
      <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-8 text-center">
        <p className="text-gray-500 dark:text-gray-400">
          No causal analysis data available. Causal analysis may still be processing.
        </p>
      </div>
    )
  }
  
  return (
    <div className="w-full h-96 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900">
      <div className="p-2 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800">
        <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
          Causal DAG: {bottleneckName || 'Bottleneck Analysis'}
        </h4>
        <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
          Red edges = increases wait time | Green edges = decreases wait time
        </p>
      </div>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        fitView
        attributionPosition="bottom-left"
      >
        <Background />
        <Controls />
        <MiniMap
          nodeColor={(node) => {
            if (node.id === 'outcome') return '#3b82f6'
            if (node.id === 'counterfactual') return '#a855f7'
            return '#94a3b8'
          }}
          maskColor="rgba(0, 0, 0, 0.1)"
        />
      </ReactFlow>
    </div>
  )
}
