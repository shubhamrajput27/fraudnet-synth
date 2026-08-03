import { useEffect, useRef, useState } from "react";
import { GATEWAY_WS_URL } from "./api.js";

// Live FL round charts (CLAUDE.md Tier 1: "live FL round charts (WebSocket)"). Connects to
// gateway/'s /ws endpoint for a given run_id and accumulates round_metric/client_metric messages
// as they arrive, so a chart can just re-render off this hook's state.
export function useLiveRun(runId) {
  const [roundMetrics, setRoundMetrics] = useState([]);
  const [clientMetrics, setClientMetrics] = useState([]);
  const [status, setStatus] = useState(runId ? "running" : "idle");
  const [finalMetrics, setFinalMetrics] = useState(null);
  const wsRef = useRef(null);

  useEffect(() => {
    if (!runId) return;
    setRoundMetrics([]);
    setClientMetrics([]);
    setStatus("running");
    setFinalMetrics(null);

    const token = sessionStorage.getItem("token");
    const ws = new WebSocket(`${GATEWAY_WS_URL}/ws?run_id=${runId}&token=${token}`);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === "round_metric") {
        setRoundMetrics((prev) => [...prev, msg.data]);
      } else if (msg.type === "client_metric") {
        setClientMetrics((prev) => [...prev, msg.data]);
      } else if (msg.type === "done") {
        setStatus(msg.status);
        setFinalMetrics(msg.final_metrics);
      }
    };
    ws.onerror = () => setStatus("failed");

    return () => ws.close();
  }, [runId]);

  return { roundMetrics, clientMetrics, status, finalMetrics };
}
