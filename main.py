from inspect_ai.util import sandbox

from smolagents.agents import ActionStep
import json
from examples.open_deep_research.run import create_agent, BROWSER_CONFIG

from typing import Any
import os

def save_agent_steps(agent, kwargs, response, sample):
    for step in agent.memory.steps:
        if isinstance(step, ActionStep):
            step.agent_memory = None
    intermediate_steps = str(agent.memory.steps)
    with open("steps.json", "w") as f:
        json.dump({
            "agent_args": kwargs,
            "intermediate_steps": intermediate_steps,
            "response": str(response),
            "sample": sample
            }, f, indent=2)

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
        
    if 'gemini' in kwargs['model_name']:
        model_params['model_id'] = kwargs['model_name'].replace('gemini/', 'openai/')
        model_params['api_key'] = os.getenv('GEMINI_API_KEY')
        model_params['api_base'] = "https://generativelanguage.googleapis.com/v1beta/openai/"
        
    if 'together_ai' in kwargs['model_name']:
        model_params['model_id'] = kwargs['model_name'].replace('together_ai/', 'openai/')
        model_params['api_key'] = os.environ.get("TOGETHERAI_API_KEY")
        model_params['api_base'] = "https://api.together.xyz/v1"
        
    agent = create_agent(model_params=model_params)
        
    response = agent.run(task['Question'])
    
    save_agent_steps(agent, kwargs, response, task)
    
    return {task_id: response}
