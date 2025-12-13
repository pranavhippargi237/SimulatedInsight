import React, { useMemo } from 'react'
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  MarkerType
} from 'reactflow'
import 'reactflow/dist/style.css'

export default function PatientFlowSankey({ flowData, bottleneckStage }) {
  // Parse flow data into nodes and edges
  const { nodes, edges } = useMemo(() => {
    if (!flowData || !flowData.stages) {
      return { nodes: [], edges: [] }
    }

    const stages = flowData.stages || []
    const nodes = []
    const edges = []
    
    // Define stage positions (horizontal flow)
    const stagePositions = {
      'arrival': { x: 0, y: 200 },
      'triage': { x: 200, y: 200 },
      'doctor': { x: 400, y: 200 },
      'labs': { x: 600, y: 100 },
      'imaging': { x: 600, y: 200 },
      'bed': { x: 600, y: 300 },
      'discharge': { x: 800, y: 150 },
      'admit': { x: 800, y: 250 },
      'lwbs': { x: 800, y: 350 }
    }

    // Create nodes for each stage
    stages.forEach((stage, idx) => {
      const stageName = stage.name || stage.stage
      const position = stagePositions[stageName] || { x: idx * 200, y: 200 }
      const patientCount = stage.patient_count || 0
      const avgWait = stage.avg_wait_minutes || 0
      const isBottleneck = bottleneckStage === stageName

      nodes.push({
        id: stageName,
        type: 'default',
        data: {
          label: (
            <div className="text-center">
              <div className="font-semibold text-sm">{stageName.toUpperCase()}</div>
              <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                {patientCount} patients
              </div>
              {avgWait > 0 && (
                <div className="text-xs text-orange-600 dark:text-orange-400 mt-1">
                  Avg wait: {avgWait.toFixed(1)} min
                </div>
              )}
            </div>
          ),
        },
        position,
        style: {
          background: isBottleneck ? '#fee2e2' : '#dbeafe',
          border: isBottleneck ? '3px solid #ef4444' : '2px solid #3b82f6',
          borderRadius: '12px',
          padding: '15px',
          minWidth: '120px',
          fontWeight: isBottleneck ? 'bold' : 'normal',
        },
      })
    })

    // Create edges based on flow transitions
    const transitions = flowData.transitions || []
    transitions.forEach((transition, idx) => {
      const from = transition.from
      const to = transition.to
      const count = transition.count || 0
      const percentage = transition.percentage || 0

      if (from && to && count > 0) {
        edges.push({
          id: `edge-${from}-${to}-${idx}`,
          source: from,
          target: to,
          type: 'smoothstep',
          animated: true,
          style: {
            stroke: transition.is_bottleneck ? '#ef4444' : '#3b82f6',
            strokeWidth: Math.max(2, Math.min(count / 10, 8)),
          },
          markerEnd: {
            type: MarkerType.ArrowClosed,
            color: transition.is_bottleneck ? '#ef4444' : '#3b82f6',
          },
          label: `${count} (${percentage.toFixed(0)}%)`,
          labelStyle: {
            fill: transition.is_bottleneck ? '#ef4444' : '#3b82f6',
            fontWeight: 600,
            fontSize: '11px',
            backgroundColor: 'white',
            padding: '2px 4px',
            borderRadius: '4px',
          },
        })
      }
    })

    return { nodes, edges }
  }, [flowData, bottleneckStage])

  const [nodesState, setNodes, onNodesChange] = useNodesState(nodes)
  const [edgesState, setEdges, onEdgesChange] = useEdgesState(edges)

  React.useEffect(() => {
    setNodes(nodes)
    setEdges(edges)
  }, [nodes, edges, setNodes, setEdges])

  if (!flowData || nodes.length === 0) {
    return (
      <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-8 text-center">
        <p className="text-gray-500 dark:text-gray-400">
          No patient flow data available.
        </p>
      </div>
    )
  }

  return (
    <div className="w-full h-96 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900">
      <div className="p-2 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800">
        <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
          Patient Flow Cascade: {bottleneckStage ? `${bottleneckStage.toUpperCase()} Bottleneck` : 'All Stages'}
        </h4>
        <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
          Red nodes/edges = bottleneck stage | Edge width = patient volume
        </p>
      </div>
      <ReactFlow
        nodes={nodesState}
        edges={edgesState}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        fitView
        attributionPosition="bottom-left"
      >
        <Background />
        <Controls />
        <MiniMap
          nodeColor={(node) => {
            if (node.id === bottleneckStage) return '#ef4444'
            return '#94a3b8'
          }}
          maskColor="rgba(0, 0, 0, 0.1)"
        />
      </ReactFlow>
    </div>
  )
}

