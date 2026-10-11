// Chart colours: the validated categorical palette (dataviz reference), light and dark steps.
// Recharts draws SVG attributes, which can't read CSS variables, so the active palette is chosen
// in JS and follows the operating system's light/dark setting.
import { useEffect, useState } from "react";

const LIGHT = {
  series: ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
  surface: "#fcfcfb", ink: "#0b0b0b", ink2: "#52514e", muted: "#898781", grid: "#e1e0d9", axis: "#c3c2b7",
};
const DARK = {
  series: ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
  surface: "#1a1a19", ink: "#ffffff", ink2: "#c3c2b7", muted: "#898781", grid: "#2c2c2a", axis: "#383835",
};

export function usePalette() {
  const mq = window.matchMedia("(prefers-color-scheme: dark)");
  const [dark, setDark] = useState(mq.matches);
  useEffect(() => {
    const on = (e) => setDark(e.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, [mq]);
  return dark ? DARK : LIGHT;
}

// Fixed colour roles, so a colour always means the same thing on every page.
export const ROLE = {
  real: 0,          // real-only arms: blue
  augmented: 2,     // real + validated synthetic: aqua
  banks: { bank_a: 0, bank_b: 1, bank_c: 2, bank_d: 3 },
};

export const axisProps = (p) => ({
  stroke: p.axis, tick: { fill: p.muted, fontSize: 12 }, tickLine: false,
});
