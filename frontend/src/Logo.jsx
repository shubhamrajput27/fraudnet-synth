// The FraudNet-Synth mark: a shield (fraud protection) around one server node joined to four banks.
// The same drawing is the browser-tab icon in index.html.
export default function Logo({ size = 32 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true" className="logo">
      <path d="M16 2 28 6v9c0 7.5-5.2 13-12 15C9.2 28 4 22.5 4 15V6Z" fill="var(--accent)" />
      <g stroke="var(--accent-ink)" strokeWidth="1.4" opacity="0.85">
        <path d="M16 15 10.5 10M16 15l5.5-5M16 15l-5.5 6M16 15l5.5 6" />
      </g>
      <g fill="var(--accent-ink)">
        <circle cx="16" cy="15" r="2.8" />
        <circle cx="10.5" cy="10" r="1.8" /><circle cx="21.5" cy="10" r="1.8" />
        <circle cx="10.5" cy="21" r="1.8" /><circle cx="21.5" cy="21" r="1.8" />
      </g>
    </svg>
  );
}
