"use client";

import { labelColor, labelSymbol, UPGRADE_COLOR } from "@/lib/labels";

// The one badge used in both places at once: floating on the board over the
// moved piece, and inline in the move list. Same look in both, so the eye ties
// them together. An upgraded move keeps its amber ring on top of the tier color.
const DARK_TEXT = new Set(["Excellent", "Inaccuracy", "Good", "Book"]);

export default function ClassBadge({
  label,
  upgraded = false,
  size = "sm",
}: {
  label: string;
  upgraded?: boolean;
  size?: "sm" | "md";
}) {
  const color = labelColor(label);
  const px = size === "md" ? 22 : 15;
  const font = size === "md" ? 12 : 9;
  return (
    <span
      aria-hidden
      className="inline-flex shrink-0 items-center justify-center rounded-[5px] font-bold leading-none"
      style={{
        width: px,
        height: px,
        fontSize: font,
        background: color,
        color: DARK_TEXT.has(label) ? "#262421" : "#ffffff",
        boxShadow: upgraded
          ? `0 0 0 2px var(--color-app), 0 0 0 4px ${UPGRADE_COLOR}`
          : undefined,
      }}
    >
      {labelSymbol(label)}
    </span>
  );
}
