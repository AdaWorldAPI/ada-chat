"""
LadybugDB AI Flow — N8N-Style Workflow Automation
═══════════════════════════════════════════════════════════════════════════════

AI Flow provides node-based workflow automation:
    - Visual-style workflow definitions (JSON-serializable)
    - Event-driven execution
    - Integration with all LadybugDB components
    - LLM-powered decision nodes
    - Awareness-triggered workflows

Node Types:
    - Trigger: HTTP, Schedule, Awareness Change, Triangle Resonance
    - Transform: Map, Filter, Reduce, VSA Encode, Alchemy
    - Action: Store, Query, HTTP Call, LLM Call
    - Logic: If/Else, Switch, Loop, Parallel
    - Integration: DuckDB Query, LanceDB Search, Graph Query

Workflow Structure:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                           Workflow                                      │
    │  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐          │
    │  │ Trigger  │───▶│Transform │───▶│  Logic   │───▶│  Action  │          │
    │  │          │    │          │    │          │    │          │          │
    │  └──────────┘    └──────────┘    └──────────┘    └──────────┘          │
    │       │                               │              │                  │
    │       │                               ▼              │                  │
    │       │                         ┌──────────┐        │                  │
    │       │                         │ Branch A │        │                  │
    │       │                         └──────────┘        │                  │
    │       │                               │              │                  │
    │       └───────────────────────────────┴──────────────┘                  │
    │                           (Data Flow)                                   │
    └─────────────────────────────────────────────────────────────────────────┘

Born: 2026-01-17
"""

import os
import json
import asyncio
import uuid
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Union, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    httpx = None
    HAS_HTTPX = False


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_TIMEOUT = 30.0
MAX_PARALLEL = 10
MAX_LOOP_ITERATIONS = 1000


# ═══════════════════════════════════════════════════════════════════════════════
# NODE TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class NodeType(str, Enum):
    # Triggers
    TRIGGER_HTTP = "trigger.http"
    TRIGGER_SCHEDULE = "trigger.schedule"
    TRIGGER_AWARENESS = "trigger.awareness"
    TRIGGER_RESONANCE = "trigger.resonance"
    TRIGGER_MANUAL = "trigger.manual"
    
    # Transforms
    TRANSFORM_MAP = "transform.map"
    TRANSFORM_FILTER = "transform.filter"
    TRANSFORM_REDUCE = "transform.reduce"
    TRANSFORM_VSA_ENCODE = "transform.vsa_encode"
    TRANSFORM_ALCHEMY = "transform.alchemy"
    TRANSFORM_TEMPLATE = "transform.template"
    
    # Actions
    ACTION_STORE = "action.store"
    ACTION_QUERY = "action.query"
    ACTION_HTTP = "action.http"
    ACTION_LLM = "action.llm"
    ACTION_LOG = "action.log"
    
    # Logic
    LOGIC_IF = "logic.if"
    LOGIC_SWITCH = "logic.switch"
    LOGIC_LOOP = "logic.loop"
    LOGIC_PARALLEL = "logic.parallel"
    LOGIC_MERGE = "logic.merge"
    
    # Integration
    INTEGRATION_DUCKDB = "integration.duckdb"
    INTEGRATION_LANCEDB = "integration.lancedb"
    INTEGRATION_MEMGRAPH = "integration.memgraph"
    INTEGRATION_KUZU = "integration.kuzu"


class ExecutionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


# ═══════════════════════════════════════════════════════════════════════════════
# WORKFLOW NODE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WorkflowNode:
    """
    A node in a workflow.
    
    Attributes:
        id: Unique identifier
        type: Node type
        name: Human-readable name
        config: Node-specific configuration
        position: Visual position (x, y) for UI
        outputs: List of output port names
        metadata: Additional data
    """
    id: str
    type: NodeType
    name: str
    config: Dict[str, Any] = field(default_factory=dict)
    position: Tuple[int, int] = (0, 0)
    outputs: List[str] = field(default_factory=lambda: ["main"])
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "name": self.name,
            "config": self.config,
            "position": list(self.position),
            "outputs": self.outputs,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowNode":
        return cls(
            id=data["id"],
            type=NodeType(data["type"]),
            name=data["name"],
            config=data.get("config", {}),
            position=tuple(data.get("position", [0, 0])),
            outputs=data.get("outputs", ["main"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class WorkflowEdge:
    """
    An edge connecting two nodes.
    
    Attributes:
        id: Unique identifier
        source_node: Source node ID
        source_output: Output port on source (default: "main")
        target_node: Target node ID
        target_input: Input port on target (default: "main")
    """
    id: str
    source_node: str
    target_node: str
    source_output: str = "main"
    target_input: str = "main"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_node": self.source_node,
            "source_output": self.source_output,
            "target_node": self.target_node,
            "target_input": self.target_input,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowEdge":
        return cls(
            id=data["id"],
            source_node=data["source_node"],
            target_node=data["target_node"],
            source_output=data.get("source_output", "main"),
            target_input=data.get("target_input", "main"),
        )


@dataclass
class Workflow:
    """
    A complete workflow definition.
    """
    id: str
    name: str
    description: str = ""
    nodes: List[WorkflowNode] = field(default_factory=list)
    edges: List[WorkflowEdge] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
    
    def get_node(self, node_id: str) -> Optional[WorkflowNode]:
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None
    
    def get_trigger_nodes(self) -> List[WorkflowNode]:
        return [n for n in self.nodes if n.type.value.startswith("trigger.")]
    
    def get_downstream_nodes(self, node_id: str) -> List[str]:
        return [e.target_node for e in self.edges if e.source_node == node_id]
    
    def get_upstream_nodes(self, node_id: str) -> List[str]:
        return [e.source_node for e in self.edges if e.target_node == node_id]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "variables": self.variables,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Workflow":
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            nodes=[WorkflowNode.from_dict(n) for n in data.get("nodes", [])],
            edges=[WorkflowEdge.from_dict(e) for e in data.get("edges", [])],
            variables=data.get("variables", {}),
            metadata=data.get("metadata", {}),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTION CONTEXT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExecutionContext:
    """
    Context for workflow execution.
    """
    execution_id: str
    workflow: Workflow
    input_data: Dict[str, Any]
    variables: Dict[str, Any] = field(default_factory=dict)
    node_outputs: Dict[str, Any] = field(default_factory=dict)
    node_status: Dict[str, ExecutionStatus] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    
    def get_node_output(self, node_id: str, output: str = "main") -> Any:
        return self.node_outputs.get(f"{node_id}.{output}")
    
    def set_node_output(self, node_id: str, output: str, value: Any):
        self.node_outputs[f"{node_id}.{output}"] = value
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "workflow_id": self.workflow.id,
            "input_data": self.input_data,
            "variables": self.variables,
            "node_outputs": self.node_outputs,
            "node_status": {k: v.value for k, v in self.node_status.items()},
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE EXECUTORS
# ═══════════════════════════════════════════════════════════════════════════════

class NodeExecutor(ABC):
    """Base class for node executors."""
    
    @abstractmethod
    async def execute(
        self,
        node: WorkflowNode,
        context: ExecutionContext,
        input_data: Any,
    ) -> Dict[str, Any]:
        """
        Execute the node.
        
        Returns dict of {output_name: output_value}
        """
        pass


class TriggerManualExecutor(NodeExecutor):
    """Manual trigger node - just passes through input."""
    
    async def execute(self, node: WorkflowNode, context: ExecutionContext, input_data: Any) -> Dict[str, Any]:
        return {"main": input_data}


class TransformMapExecutor(NodeExecutor):
    """Map transformation node."""
    
    async def execute(self, node: WorkflowNode, context: ExecutionContext, input_data: Any) -> Dict[str, Any]:
        expression = node.config.get("expression", "item")
        
        if isinstance(input_data, list):
            # Map over list
            results = []
            for i, item in enumerate(input_data):
                local_vars = {
                    "item": item,
                    "index": i,
                    "context": context.variables,
                }
                try:
                    result = eval(expression, {"__builtins__": {}}, local_vars)
                    results.append(result)
                except:
                    results.append(item)
            return {"main": results}
        else:
            # Single item
            local_vars = {
                "item": input_data,
                "index": 0,
                "context": context.variables,
            }
            try:
                result = eval(expression, {"__builtins__": {}}, local_vars)
                return {"main": result}
            except:
                return {"main": input_data}


class TransformFilterExecutor(NodeExecutor):
    """Filter transformation node."""
    
    async def execute(self, node: WorkflowNode, context: ExecutionContext, input_data: Any) -> Dict[str, Any]:
        condition = node.config.get("condition", "True")
        
        if isinstance(input_data, list):
            results = []
            for i, item in enumerate(input_data):
                local_vars = {
                    "item": item,
                    "index": i,
                    "context": context.variables,
                }
                try:
                    if eval(condition, {"__builtins__": {}}, local_vars):
                        results.append(item)
                except:
                    pass
            return {"main": results}
        else:
            local_vars = {
                "item": input_data,
                "index": 0,
                "context": context.variables,
            }
            try:
                if eval(condition, {"__builtins__": {}}, local_vars):
                    return {"main": input_data}
                else:
                    return {"main": None}
            except:
                return {"main": input_data}


class LogicIfExecutor(NodeExecutor):
    """If/Else logic node."""
    
    async def execute(self, node: WorkflowNode, context: ExecutionContext, input_data: Any) -> Dict[str, Any]:
        condition = node.config.get("condition", "True")
        
        local_vars = {
            "input": input_data,
            "context": context.variables,
        }
        
        try:
            result = eval(condition, {"__builtins__": {}}, local_vars)
            if result:
                return {"true": input_data, "false": None}
            else:
                return {"true": None, "false": input_data}
        except:
            return {"true": input_data, "false": None}


class ActionLogExecutor(NodeExecutor):
    """Log action node."""
    
    async def execute(self, node: WorkflowNode, context: ExecutionContext, input_data: Any) -> Dict[str, Any]:
        message = node.config.get("message", "Log: {input}")
        
        formatted = message.format(input=input_data)
        print(f"[AIFlow] {node.name}: {formatted}")
        
        return {"main": input_data}


class ActionHTTPExecutor(NodeExecutor):
    """HTTP action node."""
    
    async def execute(self, node: WorkflowNode, context: ExecutionContext, input_data: Any) -> Dict[str, Any]:
        if not HAS_HTTPX:
            raise ImportError("httpx not installed")
        
        method = node.config.get("method", "GET").upper()
        url = node.config.get("url", "")
        headers = node.config.get("headers", {})
        body = node.config.get("body")
        
        # Template URL with input data
        if isinstance(input_data, dict):
            for key, value in input_data.items():
                url = url.replace(f"{{{key}}}", str(value))
        
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.request(
                method=method,
                url=url,
                headers=headers,
                json=body or input_data if method in ("POST", "PUT", "PATCH") else None,
            )
            
            try:
                result = response.json()
            except:
                result = response.text
            
            return {
                "main": result,
                "status": response.status_code,
                "headers": dict(response.headers),
            }


class TransformAlchemyExecutor(NodeExecutor):
    """Alchemy transformation node."""
    
    async def execute(self, node: WorkflowNode, context: ExecutionContext, input_data: Any) -> Dict[str, Any]:
        from .triangles import Triangle, TriangleAlchemy
        
        operation = node.config.get("operation", "advance")
        
        # Convert input to triangle if needed
        if isinstance(input_data, dict):
            triangle = Triangle.from_dict(input_data)
        elif isinstance(input_data, Triangle):
            triangle = input_data
        else:
            # Create from values
            values = input_data if isinstance(input_data, (list, tuple)) else [0.5, 0.5, 0.5]
            triangle = Triangle(
                byte0=values[0] if len(values) > 0 else 0.5,
                byte1=values[1] if len(values) > 1 else 0.5,
                byte2=values[2] if len(values) > 2 else 0.5,
            )
        
        alchemy = TriangleAlchemy()
        
        if operation == "advance":
            result = alchemy.advance(triangle)
        elif operation == "gold":
            result = alchemy.alchemize(triangle, "gold")
        elif operation == "opus":
            result = alchemy.full_opus(triangle)
        else:
            result = alchemy.alchemize(triangle, operation)
        
        return {"main": result.to_dict()}


class IntegrationDuckDBExecutor(NodeExecutor):
    """DuckDB query node."""
    
    async def execute(self, node: WorkflowNode, context: ExecutionContext, input_data: Any) -> Dict[str, Any]:
        query = node.config.get("query", "SELECT 1")
        
        # Get core from context if available
        core = context.variables.get("ladybug_core")
        if core:
            results = core.query(query)
            return {"main": results}
        
        return {"main": None, "error": "LadybugCore not available in context"}


class IntegrationLanceDBExecutor(NodeExecutor):
    """LanceDB search node."""
    
    async def execute(self, node: WorkflowNode, context: ExecutionContext, input_data: Any) -> Dict[str, Any]:
        operation = node.config.get("operation", "search")
        
        vsa = context.variables.get("lance_vsa")
        if not vsa:
            return {"main": None, "error": "LanceVSA not available in context"}
        
        if operation == "search":
            import numpy as np
            query_vector = input_data if isinstance(input_data, np.ndarray) else np.array(input_data)
            limit = node.config.get("limit", 10)
            results = vsa.search(query_vector, limit=limit)
            return {"main": results}
        
        elif operation == "store":
            import numpy as np
            vector = input_data.get("vector") if isinstance(input_data, dict) else input_data
            vector_id = input_data.get("id") if isinstance(input_data, dict) else str(uuid.uuid4())
            result_id = vsa.store(vector_id, np.array(vector))
            return {"main": {"id": result_id}}
        
        return {"main": None}


# ═══════════════════════════════════════════════════════════════════════════════
# AI FLOW ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class AIFlow:
    """
    AI Flow workflow execution engine.
    
    Provides:
    - Workflow definition and storage
    - Async execution
    - Node type registry
    - Event handling
    """
    
    def __init__(self):
        self._workflows: Dict[str, Workflow] = {}
        self._executors: Dict[NodeType, NodeExecutor] = {}
        self._event_handlers: Dict[str, List[Callable]] = {}
        
        self._register_default_executors()
    
    def _register_default_executors(self):
        """Register built-in node executors."""
        self._executors = {
            NodeType.TRIGGER_MANUAL: TriggerManualExecutor(),
            NodeType.TRANSFORM_MAP: TransformMapExecutor(),
            NodeType.TRANSFORM_FILTER: TransformFilterExecutor(),
            NodeType.TRANSFORM_ALCHEMY: TransformAlchemyExecutor(),
            NodeType.LOGIC_IF: LogicIfExecutor(),
            NodeType.ACTION_LOG: ActionLogExecutor(),
            NodeType.ACTION_HTTP: ActionHTTPExecutor(),
            NodeType.INTEGRATION_DUCKDB: IntegrationDuckDBExecutor(),
            NodeType.INTEGRATION_LANCEDB: IntegrationLanceDBExecutor(),
        }
    
    def register_executor(self, node_type: NodeType, executor: NodeExecutor):
        """Register a custom node executor."""
        self._executors[node_type] = executor
    
    def register_workflow(self, workflow: Workflow):
        """Register a workflow."""
        self._workflows[workflow.id] = workflow
    
    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get a workflow by ID."""
        return self._workflows.get(workflow_id)
    
    def list_workflows(self) -> List[Dict[str, Any]]:
        """List all registered workflows."""
        return [
            {"id": w.id, "name": w.name, "description": w.description}
            for w in self._workflows.values()
        ]
    
    async def execute(
        self,
        workflow_id: str,
        input_data: Optional[Dict[str, Any]] = None,
        variables: Optional[Dict[str, Any]] = None,
    ) -> ExecutionContext:
        """
        Execute a workflow.
        
        Args:
            workflow_id: Workflow to execute
            input_data: Initial input data
            variables: Context variables (e.g., LadybugDB components)
        
        Returns:
            ExecutionContext with results
        """
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        context = ExecutionContext(
            execution_id=str(uuid.uuid4()),
            workflow=workflow,
            input_data=input_data or {},
            variables={**workflow.variables, **(variables or {})},
            started_at=datetime.now(timezone.utc),
        )
        
        try:
            # Find trigger nodes
            triggers = workflow.get_trigger_nodes()
            if not triggers:
                raise ValueError("Workflow has no trigger nodes")
            
            # Execute from first trigger
            await self._execute_node(triggers[0], context, input_data)
            
            context.completed_at = datetime.now(timezone.utc)
            
        except Exception as e:
            context.error = str(e)
            context.completed_at = datetime.now(timezone.utc)
        
        return context
    
    async def _execute_node(
        self,
        node: WorkflowNode,
        context: ExecutionContext,
        input_data: Any,
    ):
        """Execute a single node and its downstream nodes."""
        context.node_status[node.id] = ExecutionStatus.RUNNING
        
        try:
            executor = self._executors.get(node.type)
            if not executor:
                raise ValueError(f"No executor for node type: {node.type}")
            
            # Execute node
            outputs = await executor.execute(node, context, input_data)
            
            # Store outputs
            for output_name, output_value in outputs.items():
                context.set_node_output(node.id, output_name, output_value)
            
            context.node_status[node.id] = ExecutionStatus.COMPLETED
            
            # Execute downstream nodes
            for edge in context.workflow.edges:
                if edge.source_node == node.id:
                    downstream = context.workflow.get_node(edge.target_node)
                    if downstream:
                        output_value = outputs.get(edge.source_output)
                        if output_value is not None:  # Skip if None (e.g., false branch)
                            await self._execute_node(downstream, context, output_value)
            
        except Exception as e:
            context.node_status[node.id] = ExecutionStatus.FAILED
            raise
    
    # ═══════════════════════════════════════════════════════════════════════════
    # WORKFLOW BUILDER HELPERS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def create_workflow(
        self,
        name: str,
        description: str = "",
    ) -> "WorkflowBuilder":
        """Create a new workflow using builder pattern."""
        return WorkflowBuilder(name, description)


class WorkflowBuilder:
    """Builder for creating workflows."""
    
    def __init__(self, name: str, description: str = ""):
        self._id = f"wf_{uuid.uuid4().hex[:8]}"
        self._name = name
        self._description = description
        self._nodes: List[WorkflowNode] = []
        self._edges: List[WorkflowEdge] = []
        self._variables: Dict[str, Any] = {}
        self._position_x = 100
        self._position_y = 100
        self._last_node_id: Optional[str] = None
    
    def add_node(
        self,
        node_type: NodeType,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        connect_from: Optional[str] = None,
        output_port: str = "main",
    ) -> "WorkflowBuilder":
        """Add a node to the workflow."""
        node_id = f"node_{len(self._nodes) + 1}"
        
        node = WorkflowNode(
            id=node_id,
            type=node_type,
            name=name,
            config=config or {},
            position=(self._position_x, self._position_y),
        )
        self._nodes.append(node)
        
        # Auto-connect from last node or specified node
        source_node = connect_from or self._last_node_id
        if source_node:
            edge = WorkflowEdge(
                id=f"edge_{len(self._edges) + 1}",
                source_node=source_node,
                target_node=node_id,
                source_output=output_port,
            )
            self._edges.append(edge)
        
        self._last_node_id = node_id
        self._position_x += 200
        
        return self
    
    def add_trigger(self, trigger_type: str = "manual", config: Optional[Dict[str, Any]] = None) -> "WorkflowBuilder":
        """Add a trigger node."""
        type_map = {
            "manual": NodeType.TRIGGER_MANUAL,
            "http": NodeType.TRIGGER_HTTP,
            "awareness": NodeType.TRIGGER_AWARENESS,
            "schedule": NodeType.TRIGGER_SCHEDULE,
        }
        return self.add_node(type_map.get(trigger_type, NodeType.TRIGGER_MANUAL), "Trigger", config)
    
    def add_transform(self, transform_type: str, config: Dict[str, Any]) -> "WorkflowBuilder":
        """Add a transform node."""
        type_map = {
            "map": NodeType.TRANSFORM_MAP,
            "filter": NodeType.TRANSFORM_FILTER,
            "alchemy": NodeType.TRANSFORM_ALCHEMY,
        }
        return self.add_node(type_map.get(transform_type, NodeType.TRANSFORM_MAP), f"Transform: {transform_type}", config)
    
    def add_action(self, action_type: str, config: Optional[Dict[str, Any]] = None) -> "WorkflowBuilder":
        """Add an action node."""
        type_map = {
            "log": NodeType.ACTION_LOG,
            "http": NodeType.ACTION_HTTP,
            "llm": NodeType.ACTION_LLM,
        }
        return self.add_node(type_map.get(action_type, NodeType.ACTION_LOG), f"Action: {action_type}", config)
    
    def add_if(self, condition: str) -> "WorkflowBuilder":
        """Add an if/else node."""
        return self.add_node(NodeType.LOGIC_IF, "If", {"condition": condition})
    
    def set_variable(self, key: str, value: Any) -> "WorkflowBuilder":
        """Set a workflow variable."""
        self._variables[key] = value
        return self
    
    def build(self) -> Workflow:
        """Build the workflow."""
        return Workflow(
            id=self._id,
            name=self._name,
            description=self._description,
            nodes=self._nodes,
            edges=self._edges,
            variables=self._variables,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PREDEFINED WORKFLOWS
# ═══════════════════════════════════════════════════════════════════════════════

def create_awareness_pipeline() -> Workflow:
    """
    Create a standard awareness processing pipeline.
    
    Trigger → Log → Alchemy → Store
    """
    builder = WorkflowBuilder("Awareness Pipeline", "Process and transform awareness moments")
    
    return builder \
        .add_trigger("manual") \
        .add_action("log", {"message": "Processing awareness: {input}"}) \
        .add_transform("alchemy", {"operation": "advance"}) \
        .add_action("log", {"message": "Transformed to: {input}"}) \
        .build()


def create_resonance_detector() -> Workflow:
    """
    Create a resonance detection workflow.
    
    Trigger → Filter (high resonance) → Log → Store
    """
    builder = WorkflowBuilder("Resonance Detector", "Detect high-resonance triangles")
    
    return builder \
        .add_trigger("manual") \
        .add_transform("filter", {"condition": "item.get('resonance_score', 0) >= 0.7"}) \
        .add_action("log", {"message": "High resonance detected: {input}"}) \
        .build()


def create_alchemy_opus() -> Workflow:
    """
    Create a full Magnum Opus alchemy workflow.
    
    Trigger → Calcinate → Dissolve → Separate → Ferment → Distill → Coagulate → Gold
    """
    builder = WorkflowBuilder("Magnum Opus", "Full alchemical transformation to gold")
    
    return builder \
        .add_trigger("manual") \
        .add_transform("alchemy", {"operation": "calcinate"}) \
        .add_transform("alchemy", {"operation": "dissolve"}) \
        .add_transform("alchemy", {"operation": "separate"}) \
        .add_transform("alchemy", {"operation": "ferment"}) \
        .add_transform("alchemy", {"operation": "distill"}) \
        .add_transform("alchemy", {"operation": "coagulate"}) \
        .add_transform("alchemy", {"operation": "gold"}) \
        .add_action("log", {"message": "Transmutation complete: {input}"}) \
        .build()
