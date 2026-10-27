// useRunStream(jobId): live view of one run through the gateway's WebSocket (/ws/runs/:jobId).
// The gateway first replays what MongoDB already holds (catch-up), then streams new events, so
// opening a finished run shows its full history and opening a running one shows it growing.
import { useEffect, useReducer } from "react";
import { api, liveUrl } from "./api.js";
import { empty, reducer } from "./runState.js";

export function useRunStream(jobId) {
  const [state, dispatch] = useReducer(reducer, empty);
  useEffect(() => {
    if (!jobId) return undefined;
    dispatch({ type: "reset" });
    let closed = false;
    api(`/runs/${encodeURIComponent(jobId)}/models`)
      .then((models) => !closed && dispatch({ type: "models_loaded", models }))
      .catch(() => {});
    const ws = new WebSocket(liveUrl(jobId));
    ws.onmessage = (m) => dispatch(JSON.parse(m.data));
    ws.onerror = () => !closed && dispatch({ type: "error", error: "live connection failed" });
    return () => { closed = true; ws.close(); };
  }, [jobId]);
  return state;
}
