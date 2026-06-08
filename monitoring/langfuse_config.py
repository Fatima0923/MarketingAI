# monitoring/langfuse_config.py
import os, time, functools
from typing import Optional, Dict, Any
from datetime import datetime

_client  = None
_enabled = False

def init_langfuse(public_key: str = "", secret_key: str = "", host: str = ""):
    global _client, _enabled
    pk = public_key or os.getenv("LANGFUSE_PUBLIC_KEY", "")
    sk = secret_key or os.getenv("LANGFUSE_SECRET_KEY", "")
    h  = host       or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    if not pk or not sk:
        print("[Langfuse] Keys not set — observability disabled")
        _enabled = False
        return None
    try:
        from langfuse import Langfuse
        _client  = Langfuse(public_key=pk, secret_key=sk, host=h)
        _enabled = True
        print(f"[Langfuse] Enabled — traces at {h}")
        return _client
    except Exception as e:
        print(f"[Langfuse] Init failed: {e}")
        _enabled = False
        return None

def get_client():
    global _client
    if _client is None:
        init_langfuse()
    return _client

def is_enabled(): return _enabled

class TraceContext:
    def __init__(self, name, trace_type="span", input_data=None, metadata=None):
        self.name, self.trace_type = name, trace_type
        self.input_data, self.metadata = input_data, metadata or {}
        self._output = self._error = self._span = None
        self._tokens = {}
        self._start  = time.time()

    def __enter__(self):
        if _enabled:
            try:
                c = get_client()
                if c:
                    self._span = c.trace(name=self.name, input=self.input_data,
                                         metadata=self.metadata)
            except Exception:
                pass
        return self

    def set_output(self, o): self._output = o
    def set_tokens(self, prompt=0, completion=0):
        self._tokens = {"prompt": prompt, "completion": completion}
    def set_error(self, e): self._error = e

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = round(time.time() - self._start, 3)
        if exc_type: self._error = str(exc_val)
        if _enabled and self._span:
            try:
                self._span.end(
                    output=self._output,
                    metadata={**self.metadata, "latency_s": elapsed,
                              "tokens": self._tokens, "error": self._error},
                    level="ERROR" if self._error else "DEFAULT",
                )
            except Exception:
                pass
        return False

_events = []

def log_event(event_type, name, data=None, error=None):
    e = {"timestamp": datetime.now().isoformat(), "type": event_type,
         "name": name, "data": data or {}, "error": error}
    _events.append(e)
    tag = f"[{event_type.upper()}]"
    print(f"{tag} {name}" + (f" — {error}" if error else ""))
    if _enabled:
        try:
            c = get_client()
            if c: c.event(name=name, metadata=e)
        except Exception:
            pass

def get_events(): return list(_events)

def flush():
    if _enabled:
        try:
            c = get_client()
            if c: c.flush(); print("[Langfuse] Flushed.")
        except Exception:
            pass
