// State of one live run, built from the gateway's WebSocket events (pure, unit-tested).

export const empty = { status: null, messages: [], rounds: [], models: [], banksDone: [], ended: false, error: null };

const sameModel = (a, b) => a.arm === b.arm && a.seed === b.seed && a.model === b.model;

export function reducer(state, ev) {
  switch (ev.type) {
    case "reset": return { ...empty };
    case "status": return { ...state, status: ev.status };
    case "message": return { ...state, messages: [...state.messages, ev.message].slice(-200) };
    case "round": {
      const key = `${ev.arm}|${ev.seed}|${ev.round}`;
      if (state.rounds.some((r) => `${r.arm}|${r.seed}|${r.round}` === key)) return state;   // no duplicates
      return { ...state, rounds: [...state.rounds, ev] };
    }
    case "model_done": return { ...state, models: [...state.models.filter((m) => !sameModel(m, ev)), ev] };
    case "models_loaded": return { ...state, models: [...ev.models,
      ...state.models.filter((m) => !ev.models.some((x) => sameModel(x, m)))] };
    case "bank_done": return { ...state, banksDone: [...state.banksDone, ev] };
    case "end": return { ...state, status: ev.status, ended: true };
    case "error": return { ...state, error: ev.error };
    default: return state;
  }
}
