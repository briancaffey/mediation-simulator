from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig
from langchain_core.messages import HumanMessage


class GetMediationCase(FunctionBaseConfig, name="get_mediation_case"):
    """
    Function for getting the case state from the redis memory
    """
    pass


@register_function(config_type=GetMediationCase)
async def get_mediation_case(config: GetMediationCase, builder: Builder):

    memory = builder.get_memory_client("redis_memory")

    async def _get_mediation_case(unused: str) -> dict:

        # get information from the request using the aiq context
        from aiq.builder.context import AIQContext
        aiq_context = AIQContext.get()
        path_params: dict[str, str] | None = aiq_context.metadata.path_params

        case_id = path_params.get("case_id")
        case_state = await memory.get_case_state(case_id)
        return dict(case_state)

    yield FunctionInfo.from_fn(
        _get_mediation_case,
        description="Returns the acquired user defined request attriubutes."
    )
