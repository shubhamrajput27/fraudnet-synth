// useRunStream(jobId): live view of one run through the gateway's WebSocket (/ws/runs/:jobId).
// The gateway first replays what MongoDB already holds (catch-up), then streams new events, so
// opening a finished run shows its full history and opening a running one shows it growing.
// If the connection drops before the run ends (Wi-Fi blip, gateway restart), it reconnects: the
// catch-up replays the history and the reducer drops rounds it already has, so nothing is doubled.
import { useEffect, useReducer } from "react";
import { api, liveUrl } from "./api.js";
import { empty, reducer } from "./runState.js";

const RETRY_MS = [1000, 2000, 4000, 8000, 15000];   // then every 15 s

export function useRunStream(jobId) {
  const [state, dispatch] = useReducer(reducer, empty);
  useEffect(() => {
    if (!jobId) return undefined;
    dispatch({ type: "reset" });
    let closed = false;       // the page left this run: stop for good
    let ended = false;        // the run finished: a closed socket is expected
    let attempt = 0;
    let ws = null;
    let timer = null;
    api(`/runs/${encodeURIComponent(jobId)}/models`)
      .then((models) => !closed && dispatch({ type: "models_loaded", models }))
      .catch(() => {});
    const connect = () => {
      ws = new WebSocket(liveUrl(jobId));
      ws.onopen = () => { attempt = 0; dispatch({ type: "error", error: null }); };
      ws.onmessage = (m) => {
        const ev = JSON.parse(m.data);
        if (ev.type === "end") ended = true;
        dispatch(ev);
      };
      ws.onclose = () => {
        if (closed || ended) return;
        const wait = RETRY_MS[Math.min(attempt++, RETRY_MS.length - 1)];
        dispatch({ type: "error", error: `live connection lost, reconnecting in ${wait / 1000} s…` });
        timer = setTimeout(connect, wait);
      };
    };
    connect();
    return () => { closed = true; clearTimeout(timer); ws?.close(); };
  }, [jobId]);
  return state;
}
