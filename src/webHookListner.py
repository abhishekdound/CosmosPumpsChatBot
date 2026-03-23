from .dataAcquisition import DataAcquisition
import threading

retriever_lock = threading.Lock()
da = DataAcquisition()
session_retrievers: dict = {}






def register_session(session_id: str):
    """Register session on chat start — copies any pending crawl data."""
    with retriever_lock:
        already_registered = session_id in session_retrievers

    if not already_registered:
        retriever = da.build_retriever(session_id)
        with retriever_lock:
            session_retrievers[session_id] = retriever
        print(f"[SESSION] Registered {session_id}")


def get_retriever_for_session(session_id: str):
    with retriever_lock:
        retriever = session_retrievers.get(session_id)

    if retriever is None:
        retriever = da.build_retriever(session_id)
        with retriever_lock:
            session_retrievers[session_id] = retriever

    return retriever


def set_retriever_for_session(session_id: str, retriever):
    with retriever_lock:
        session_retrievers[session_id] = retriever


def clear_session(session_id: str):
    with retriever_lock:
        session_retrievers.pop(session_id, None)
    da.clear_session(session_id)
    print(f"[SESSION] Cleaned up {session_id}")





