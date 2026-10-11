// useRunStream(jobId): live view of one run through the gateway's WebSocket (/ws/runs/:jobId).
// The gateway first replays what MongoDB already holds (catch-up), then streams new events, so
// opening a finished run shows its full history and opening a running one shows it growing.
import { useEffect, useReducer } from "react";
import { api, liveUrl } from "./api.js";

const empty = { status: null, messages: [], rounds: [], models: [], banksDone: [], ended: false, error: null };

function reducer(state, ev) {
  switch (ev.type) {
    case "reset": return { ...empty };
    case "status": return { ...state, status: ev.status };
    case "message": return { ...state, messages: [...state.messages, ev.message].slice(-200) };
    case "round": {
      const key = `${ev.arm}|${ev.seed}|${ev.round}`;
      if (state.rounds.some((r) => `${r.arm}|${r.seed}|${r.round}` === key)) return state;   // no duplicates
      return { ...state, rounds: [...state.rounds, ev] };
    }
    case "model_done": {
      const others = state.models.filter((m) => !(m.arm === ev.arm && m.seed === ev.seed && m.model === ev.model));
      return { ...state, models: [...others, ev] };
    }
    case "models_loaded": return { ...state, models: [...ev.models, ...state.models.filter(
      (m) => !ev.models.some((x) => x.arm === m.arm && x.seed === m.seed && x.model === m.model))] };
    case "bank_done": return { ...state, banksDone: [...state.banksDone, ev] };
    case "end": return { ...state, status: ev.status, ended: true };
    case "error": return { ...state, error: ev.error };
    default: return state;
  }
}

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
