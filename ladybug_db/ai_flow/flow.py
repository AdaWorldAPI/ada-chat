from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime

class Flow:
    def __init__(self, name: str):
        self.id = str(uuid.uuid4())
        self.name = name
        self.steps = []
        self.context = {}

    def add_step(self, step_func):
        self.steps.append(step_func)
        return self

    async def run(self, initial_context: Dict[str, Any] = None):
        self.context = initial_context or {}
        self.context['flow_id'] = self.id
        self.context['started_at'] = datetime.now().isoformat()
        
        for step in self.steps:
            try:
                result = await step(self.context)
                self.context.update(result or {})
            except Exception as e:
                self.context['error'] = str(e)
                break
        
        self.context['completed_at'] = datetime.now().isoformat()
        return self.context

class Agent:
    def __init__(self, name: str, model: str = "grok-beta"):
        self.name = name
        self.model = model
    
    async def process(self, input_data: Any) -> Dict[str, Any]:
        # Placeholder for LLM processing
        return {"response": f"Processed by {self.name}"}
