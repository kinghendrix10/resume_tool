"use client";

import { useMemo } from "react";
import {
  Background,
  Controls,
  MiniMap,
  ReactFlow,
  useEdgesState,
  useNodesState,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import type { ResumePayload } from "@/lib/resume/schema";
import { educationGraphEdges } from "@/lib/resume/chart-data";

type Graph = ReturnType<typeof educationGraphEdges>;

function EduFlowView({ graph }: { graph: Graph }) {
  const initialNodes = useMemo(
    () =>
      graph.nodes.map((n, i) => ({
        id: n.id,
        position: { x: (i % 3) * 180, y: Math.floor(i / 3) * 100 },
        data: { label: n.label },
        type: "default" as const,
      })),
    [graph.nodes]
  );
  const initialEdges = useMemo(
    () =>
      graph.edges.map((e, i) => ({
        id: `e-${i}`,
        source: e.source,
        target: e.target,
        animated: true,
      })),
    [graph.edges]
  );

  const [nodes, , onNodesChange] = useNodesState(initialNodes);
  const [edges, , onEdgesChange] = useEdgesState(initialEdges);

  return (
    <div className="h-full w-full">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        fitView
      >
        <MiniMap />
        <Controls />
        <Background gap={12} size={1} />
      </ReactFlow>
    </div>
  );
}

export function EducationMiniFlow({ resume }: { resume: ResumePayload }) {
  const graph = useMemo(() => educationGraphEdges(resume), [resume]);
  const key = graph.nodes.map((n) => n.id).join("|");
  return <EduFlowView key={key} graph={graph} />;
}

