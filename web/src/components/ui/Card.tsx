import { type HTMLAttributes, forwardRef } from "react";
import { cn } from "../../lib/cn";

export interface CardProps extends HTMLAttributes<HTMLDivElement> {
  /** Adds pointer affordance + hover lift. */
  interactive?: boolean;
}

const Card = forwardRef<HTMLDivElement, CardProps>(
  ({ className, interactive = false, ...props }, ref) => (
    <div
      ref={ref}
      className={cn(
        "bg-surface rounded-lg shadow-card overflow-hidden transition-[transform,box-shadow] duration-150",
        interactive &&
          "cursor-pointer hover:-translate-y-0.5 hover:shadow-[var(--shadow-card-hover)]",
        className,
      )}
      {...props}
    />
  ),
);
Card.displayName = "Card";

export default Card;
