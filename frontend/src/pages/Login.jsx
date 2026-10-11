import { useState } from "react";
import { login } from "../api.js";

export default function Login() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  async function submit(e) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try { await login(username, password); } catch (err) { setError(err.message); } finally { setBusy(false); }
  }

  return (
    <div className="login">
      <form className="card" onSubmit={submit}>
        <div>
          <h1>FraudNet-Synth</h1>
          <p className="muted">Privacy-preserving fraud detection with federated learning and synthetic data</p>
        </div>
        <div className="field">
          <label htmlFor="u">Username</label>
          <input id="u" type="text" autoComplete="username" value={username} onChange={(e) => setUsername(e.target.value)} required />
        </div>
        <div className="field">
          <label htmlFor="p">Password</label>
          <input id="p" type="password" autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)} required />
        </div>
        {error && <p className="error" role="alert">{error}</p>}
        <button className="btn primary" disabled={busy}>{busy ? "Signing in…" : "Sign in"}</button>
        <p className="note">Accounts: admin · bank_a … bank_d (bank operators) · guide (viewer).</p>
      </form>
    </div>
  );
}
