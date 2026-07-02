import { type ButtonHTMLAttributes, forwardRef } from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "../../lib/cn";

const button = cva(
  "inline-flex items-center justify-center gap-2 rounded-[var(--radius)] font-ui font-semibold " +
    "transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-50 " +
    "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent",
  {
    variants: {
      variant: {
        primary: "bg-accent text-white hover:bg-accent-hover disabled:bg-sand disabled:text-muted",
        secondary: "bg-sand text-ink border border-edge hover:border-accent",
        ghost: "text-muted hover:text-ink hover:bg-sand",
      },
      size: {
        sm: "text-[13px] px-3 py-1.5",
        md: "text-sm px-4 py-2",
      },
    },
    defaultVariants: { variant: "primary", size: "md" },
  },
);

export interface ButtonProps
  extends ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof button> {}

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, ...props }, ref) => (
    <button ref={ref} className={cn(button({ variant, size }), className)} {...props} />
  ),
);
Button.displayName = "Button";

export default Button;
