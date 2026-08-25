"""
Compatibility shim — must be imported before anything that imports `ragas`.

`ragas` (both 0.4.3 and 0.3.9) hardcodes:
    from langchain_community.chat_models.vertexai import ChatVertexAI
at module-import time, unconditionally — even for users who never touch
VertexAI. That submodule was removed from langchain-community; the real
class now lives in `langchain_google_vertexai`. See ragas issue #2745.

This registers a fake `langchain_community.chat_models.vertexai` module in
sys.modules that re-exports the real, working `ChatVertexAI` from its new
location — so ragas's stale import path resolves to the actual class
(costs ~8s once per process, one-time SDK import), not a fake stub.
"""
import sys

from langchain_google_vertexai import ChatVertexAI

_vertexai_shim = type(sys)("langchain_community.chat_models.vertexai")
_vertexai_shim.ChatVertexAI = ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_shim
