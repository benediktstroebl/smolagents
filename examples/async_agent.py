import math
import asyncio
import os

from dotenv import load_dotenv

from smolagents import ToolCallingAgent, CodeAgent
from smolagents.models import LiteLLMModel
from smolagents import Tool


load_dotenv()


class CalculatePowerTool(Tool):
    name = "calculate_power"
    description = "Calculates the power of a number."
    inputs = {
        "base": {"type": "number", "description": "The base number."},
        "exponent": {"type": "number", "description": "The exponent number."}
    }
    output_type = "number"

    async def forward(self, base: float, exponent: float) -> float:
        await asyncio.sleep(0.1)
        # Regular synchronous implementation
        return math.pow(base, exponent)


async def main():
    # Initialize the OpenAI model
    model = LiteLLMModel(
        model_id="openai/gpt-4o-mini",
    )
    
    # Create the agent with the calculation tool
    tools = [
        CalculatePowerTool()
    ]
    agent = ToolCallingAgent(tools=tools, model=model)
    
    # Run the agent synchronously but in an async context
    result = await agent.arun("What is the result of 2 power 3.7384?")
    print("Result:", result)
    

if __name__ == "__main__":
    asyncio.run(main())