# borrowed from https://github.com/NVIDIA/AIQToolkit/blob/develop/examples/simple_calculator/src/aiq_simple_calculator/register.py

import logging

# Configure logging to suppress warnings
logging.getLogger("aiq.data_models.discovery_metadata").setLevel(logging.ERROR)
logging.getLogger("aiq.data_models.discovery_metadata").propagate = False
logging.getLogger("aiq.agent.register").setLevel(logging.ERROR)
logging.getLogger("aiq.agent.register").propagate = False
logging.getLogger("aiq.runtime.loader").setLevel(logging.ERROR)
logging.getLogger("aiq.runtime.loader").propagate = False

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class InequalityToolConfig(FunctionBaseConfig, name="calculator_inequality"):
    pass


@register_function(config_type=InequalityToolConfig)
async def calculator_inequality(tool_config: InequalityToolConfig, builder: Builder):

    import re

    async def _calculator_inequality(text: str) -> str:
        numbers = re.findall(r"\d+", text)
        a = int(numbers[0])
        b = int(numbers[1])

        if a > b:
            return f"First number {a} is greater than the second number {b}"
        if a < b:
            return f"First number {a} is less than the second number {b}"

        return f"First number {a} is equal to the second number {b}"

    # Create a Generic AIQ Toolkit tool that can be used with any supported LLM framework
    yield FunctionInfo.from_fn(
        _calculator_inequality,
        description=(
            "This is a mathematical tool used to perform an inequality comparison between two numbers. "
            "It takes two numbers as an input and determines if one is greater or are equal."
        ),
    )


class MultiplyToolConfig(FunctionBaseConfig, name="calculator_multiply"):
    pass


@register_function(config_type=MultiplyToolConfig)
async def calculator_multiply(config: MultiplyToolConfig, builder: Builder):

    import re

    async def _calculator_multiply(text: str) -> str:
        numbers = re.findall(r"\d+", text)
        a = int(numbers[0])
        b = int(numbers[1])

        return f"The product of {a} * {b} is {a * b}"

    # Create a Generic AIQ Toolkit tool that can be used with any supported LLM framework
    yield FunctionInfo.from_fn(
        _calculator_multiply,
        description=(
            "This is a mathematical tool used to multiply two numbers together. "
            "It takes 2 numbers as an input and computes their numeric product as the output."
        ),
    )


class DivisionToolConfig(FunctionBaseConfig, name="calculator_divide"):
    pass


@register_function(config_type=DivisionToolConfig)
async def calculator_divide(config: DivisionToolConfig, builder: Builder):

    import re

    async def _calculator_divide(text: str) -> str:
        numbers = re.findall(r"\d+", text)
        a = int(numbers[0])
        b = int(numbers[1])

        return f"The result of {a} / {b} is {a / b}"

    # Create a Generic AIQ Toolkit tool that can be used with any supported LLM framework
    yield FunctionInfo.from_fn(
        _calculator_divide,
        description=(
            "This is a mathematical tool used to divide one number by another. "
            "It takes 2 numbers as an input and computes their numeric quotient as the output."
        ),
    )


class SubtractToolConfig(FunctionBaseConfig, name="calculator_subtract"):
    pass


@register_function(config_type=SubtractToolConfig)
async def calculator_subtract(config: SubtractToolConfig, builder: Builder):

    import re

    async def _calculator_subtract(text: str) -> str:
        numbers = re.findall(r"\d+", text)
        a = int(numbers[0])
        b = int(numbers[1])

        return f"The result of {a} - {b} is {a - b}"

    # Create a Generic AIQ Toolkit tool that can be used with any supported LLM framework
    yield FunctionInfo.from_fn(
        _calculator_subtract,
        description=(
            "This is a mathematical tool used to subtract one number from another. "
            "It takes 2 numbers as an input and computes their numeric difference as the output."
        ),
    )


class AddToolConfig(FunctionBaseConfig, name="calculator_add"):
    pass


@register_function(config_type=AddToolConfig)
async def calculator_add(config: AddToolConfig, builder: Builder):

    import re

    async def _calculator_add(text: str) -> str:
        numbers = re.findall(r"\d+", text)
        a = int(numbers[0])
        b = int(numbers[1])

        return f"The result of {a} + {b} is {a + b}"

    # Create a Generic AIQ Toolkit tool that can be used with any supported LLM framework
    yield FunctionInfo.from_fn(
        _calculator_add,
        description=(
            "This is a mathematical tool used to add two numbers. "
            "It takes 2 numbers as an input and computes their numeric sum as the output."
        ),
    )
