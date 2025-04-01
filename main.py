from inspect_ai.util import sandbox

import asyncio

from examples.open_deep_research.run import create_agent, BROWSER_CONFIG

from typing import Any
import os

async def run(sample: dict[str, Any], **kwargs) -> dict[str, Any]:
    
    os.makedirs(f"./{BROWSER_CONFIG['downloads_folder']}", exist_ok=True)
    
    agent = create_agent(model_id=kwargs['model_name'])
        
    response = await agent.arun(sample["input"][0]["content"])
    
    return {"output": response}
