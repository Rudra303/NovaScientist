from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import asyncio
from typing import Dict, Any

from novascientist.framework import NovaScientistFramework, NovaScientistConfig
from novascientist.global_state import NovaScientistState, NovaScientistStateManager

app = FastAPI(
    title="NovaScientist API",
    description="API for the NovaScientist multi-agent system",
    version="1.0.0"
)

# In-memory store of active jobs
active_jobs: Dict[str, Any] = {}

class StartResearchRequest(BaseModel):
    goal: str
    n_hypotheses: int = 4

class RunTournamentRequest(BaseModel):
    goal: str
    k_bracket: int = 8

@app.post("/api/v1/research/start")
async def start_research(request: StartResearchRequest, background_tasks: BackgroundTasks):
    """Start a new research job for a given goal."""
    if request.goal in active_jobs:
        raise HTTPException(status_code=400, detail="Research for this goal is already running.")
        
    try:
        initial_state = NovaScientistState(goal=request.goal)
    except FileExistsError:
        # Load existing state if we want to resume, but for now we'll just return an error
        raise HTTPException(status_code=400, detail="Goal directory already exists. Use resume endpoint.")

    config = NovaScientistConfig()
    state_manager = NovaScientistStateManager(initial_state)
    framework = NovaScientistFramework(config, state_manager)
    
    active_jobs[request.goal] = {
        "status": "running",
        "framework": framework
    }
    
    # Run the system in the background
    background_tasks.add_task(run_framework, request.goal, framework)
    
    return {"message": "Research started", "goal": request.goal}

async def run_framework(goal: str, framework: NovaScientistFramework):
    try:
        await framework.run()
        active_jobs[goal]["status"] = "completed"
    except Exception as e:
        active_jobs[goal]["status"] = f"failed: {str(e)}"

@app.get("/api/v1/research/status")
async def get_status(goal: str):
    """Get the status of a running research job."""
    if goal in active_jobs:
        return {"goal": goal, "status": active_jobs[goal]["status"]}
    
    # If not in active jobs, check if it exists in the output directory
    try:
        state = NovaScientistState.load_latest(goal=goal)
        if state:
            return {"goal": goal, "status": "exists_on_disk", "is_finished": state.final_report is not None}
    except Exception:
        pass
        
    raise HTTPException(status_code=404, detail="Goal not found")

@app.get("/api/v1/research/hypotheses")
async def get_hypotheses(goal: str):
    """Fetch the generated hypotheses for a specific goal."""
    state = NovaScientistState.load_latest(goal=goal)
    if not state:
        raise HTTPException(status_code=404, detail="Goal not found")
        
    return {
        "generated": [h.dict() for h in state.generated_hypotheses],
        "reviewed": [h.dict() for h in state.reviewed_hypotheses],
        "evolved": [h.dict() for h in state.evolved_hypotheses],
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
