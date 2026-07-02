import { type HTMLAttributes } from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "../../lib/cn";

const badge = cva(
  "inline-flex items-center gap-1 rounded-full border font-ui font-semibold uppercase tracking-wider " +
    "text-[11px] leading-none px-2 py-1",
  {
    variants: {
      variant: {
        success: "text-success border-success/40 bg-success/10",
        warning: "text-warning border-warning/40 bg-warning/10",
        danger: "text-danger border-danger/40 bg-danger/10",
        neutral: "text-muted border-edge bg-sand",
      },
    },
    defaultVariants: { variant: "neutral" },
  },
);

export type BadgeVariant = NonNullable<VariantProps<typeof badge>["variant"]>;

export interface BadgeProps
  extends HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badge> {}

export default function Badge({ className, variant, ...props }: BadgeProps) {
  return <span className={cn(badge({ variant }), className)} {...props} />;
}

/** Book pipeline status → badge appearance. Covers all five backend statuses. */
export function statusBadge(status: string): { label: string; variant: BadgeVariant } {
  switch (status) {
    case "ready":
      return { label: "Ready", variant: "success" };
    case "ingesting":
      return { label: "Ingesting", variant: "warning" };
    case "embedding":
      return { label: "Embedding", variant: "warning" };
    case "indexing":
      return { label: "Indexing", variant: "warning" };
    case "failed":
      return { label: "Failed", variant: "danger" };
    default:
      return { label: status, variant: "neutral" };
  }
}
