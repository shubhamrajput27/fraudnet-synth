// Everything the dashboard sends goes to the GATEWAY (/api, /ws), never to FastAPI directly.
// Login token: kept in memory and in sessionStorage, so it survives a refresh in the same tab
// and is cleared when the tab closes (decision D-054).

const TOKEN_KEY = "fns_token";
const USER_KEY = "fns_user";
let token = sessionStorage.getItem(TOKEN_KEY);
let user = JSON.parse(sessionStorage.getItem(USER_KEY) ?? "null");
const listeners = new Set();

export const getUser = () => user;
export const onAuthChange = (fn) => { listeners.add(fn); return () => listeners.delete(fn); };
const notify = () => listeners.forEach((fn) => fn(user));

export async function login(username, password) {
  const res = await fetch("/api/auth/login", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ username, password }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error ?? `login failed (${res.status})`);
  token = data.token;
  user = data.user;
  sessionStorage.setItem(TOKEN_KEY, token);
  sessionStorage.setItem(USER_KEY, JSON.stringify(user));
  notify();
  return user;
}

export function logout() {
  token = null;
  user = null;
  sessionStorage.removeItem(TOKEN_KEY);
  sessionStorage.removeItem(USER_KEY);
  notify();
}

export async function api(path, { method = "GET", body } = {}) {
  const res = await fetch(`/api${path}`, {
    method,
    headers: { Authorization: `Bearer ${token}`, ...(body !== undefined ? { "Content-Type": "application/json" } : {}) },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (res.status === 401) {          // token expired or invalid: back to the login screen
    logout();
    throw new Error("session expired, please log in again");
  }
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    const detail = data?.detail ? JSON.stringify(data.detail) : data?.error;
    throw new Error(detail ?? `request failed (${res.status})`);
  }
  return data;
}

// Downloads (export) also need the Authorization header, so we fetch a blob and save it.
export async function download(path, filename) {
  const res = await fetch(`/api${path}`, { headers: { Authorization: `Bearer ${token}` } });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.error ?? `download failed (${res.status})`);
  }
  const url = URL.createObjectURL(await res.blob());
  const a = Object.assign(document.createElement("a"), { href: url, download: filename });
  // Attached to the page and revoked a moment later: some browsers cancel the download otherwise.
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// Browsers can't put headers on WebSockets, so the gateway accepts the token as ?token= (D-049).
export function liveUrl(jobId) {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${location.host}/ws/runs/${encodeURIComponent(jobId)}?token=${encodeURIComponent(token)}`;
}
