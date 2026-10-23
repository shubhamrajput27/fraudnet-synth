"""Shared chart palette and axes style, so every saved plot in results/ looks the same."""

INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style_axes(ax, grid: str | None = None):
    """Light surface, no top/right spines, muted ticks. grid: None, "both" or "y"."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    if grid:
        ax.grid(color=GRID, linewidth=0.8, **({"axis": "y"} if grid == "y" else {}))
    ax.set_axisbelow(True)
