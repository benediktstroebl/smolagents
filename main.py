from inspect_ai.util import sandbox

import asyncio

from examples.open_deep_research.run import create_agent, BROWSER_CONFIG

from typing import Any
import os

def run(input: dict[str, dict], **kwargs) -> dict[str, str]:

    assert 'model_name' in kwargs, 'model_name is required'
    assert len(input) == 1, 'input must contain only one task'
    
    task_id, task = list(input.items())[0]
    
    os.makedirs(f"./{BROWSER_CONFIG['downloads_folder']}", exist_ok=True)
    
    model_params = {}
    model_params['model_id'] = kwargs['model_name']
    if 'reasoning_effort' in kwargs:
        model_params['reasoning_effort'] = kwargs['reasoning_effort']
    if 'temperature' in kwargs:
        model_params['temperature'] = kwargs['temperature']
    
    agent = create_agent(model_params=model_params)
        
    response = asyncio.run(agent.arun(task['Question']))
    
    return {task_id: response}
