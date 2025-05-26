from typing import Sequence, Optional
import json

from aiq.builder.builder import Builder
from aiq.cli.register_workflow import register_memory
from aiq.data_models.memory import MemoryBaseConfig
from aiq.memory.interfaces import MemoryEditor, MemoryItem
from langchain_redis import RedisChatMessageHistory
from langchain_core.messages import BaseMessage
from redis import Redis


class RedisMemoryConfig(MemoryBaseConfig, name="redis_memory"):
    connection_url: str


@register_memory(config_type=RedisMemoryConfig)
async def redis_memory(config: RedisMemoryConfig, builder: Builder):

    class RedisMemoryEditor(MemoryEditor):
        def __init__(self, config: RedisMemoryConfig):
            self._conn_url = config.connection_url
            self.redis = Redis.from_url(self._conn_url)

        # case generation state management
        async def save_case_description(
            self, case_description: str, case_id: str
        ) -> None:
            """
            sets the case description using the <case_id>_case_description as the redis key
            """
            self.redis.set(f"{case_id}_case_description", case_description)

        async def save_case_state(self, case_state: dict, case_id: str) -> None:
            """
            sets the case state using the <case_id>_case_state as the redis key
            """
            self.redis.set(f"{case_id}_case_state", json.dumps(case_state))

        async def get_case_state(self, case_id: str) -> dict:
            """
            gets the case state using the <case_id>_case_state as the redis key as json
            """
            state = self.redis.get(f"{case_id}_case_state")
            if state is None:
                return {}
            return json.loads(state)

        async def get_session_state(self, session_id: str) -> dict:
            """
            gets the session state using the <session_id>_session_state as the redis key as json
            """
            state = self.redis.get(f"session:{session_id}:session_state")
            if state is None:
                return {}
            return json.loads(state)

        async def save_session_state(self, session_state: dict, session_id: str) -> None:
            """
            saves the session state using the <session_id>_session_state as the redis key as json
            """
            self.redis.set(f"session:{session_id}:session_state", json.dumps(session_state))

        async def get_session_data(self, case_id: str, session_id: str) -> dict:
            """
            gets the session data using the <case_id>_<session_id>_session_data as the redis key as json
            """
            client = await self.get_client(session_id)
            messages = await client.aget_messages()
            if messages is None:
                return []
            return messages

        async def set_session_field(self, session_id: str, field: str, value: str) -> None:
            """
            sets the session field using the <session_id>_<field> as the redis key
            """
            self.redis.set(f"session:{session_id}:{field}", value)

        async def get_session_field(self, session_id: str, field: str) -> Optional[str]:
            """
            gets the session field using the <session_id>_<field> as the redis key
            """
            return self.redis.get(f"session:{session_id}:{field}")

        async def get_client(self, session_id: str) -> RedisChatMessageHistory:
            conn = RedisChatMessageHistory(
                session_id=session_id, redis_url=self._conn_url
            )
            return conn

        # mediation session state management
        async def add_messages(
            self, items: Sequence[BaseMessage], session_id: str
        ) -> None:
            client = await self.get_client(session_id)
            await client.aadd_messages(items)

        async def get_messages(self, session_id: str) -> Sequence[BaseMessage]:
            client = await self.get_client(session_id)
            messages = await client.aget_messages()
            return messages

        async def remove_messages(self, session_id: str) -> None:
            client = await self.get_client(session_id)
            client.clear()

        # for implementation of the interface, but not used
        async def add_items(self, items: Sequence[MemoryItem], session_id: str) -> None:
            client = await self.get_client(session_id)
            await client.aadd_messages(items)

        async def remove_items(
            self, items: Sequence[MemoryItem], session_id: str
        ) -> None:
            client = await self.get_client(session_id)
            client.remove_messages(items)

        async def search(self, query: str, top_k: int = 5) -> Sequence[MemoryItem]:
            client = await self.get_client("abc")
            messages = await client.aget_messages("abc")
            return messages

    yield RedisMemoryEditor(config)
