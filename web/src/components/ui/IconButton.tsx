import { type ButtonHTMLAttributes, forwardRef } from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "../../lib/cn";

const iconButton = cva(
  "inline-flex items-center justify-center rounded-[var(--radius)] transition-colors duration-150 " +
    "disabled:cursor-not-allowed disabled:opacity-50 " +
    "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent",
  {
    variants: {
      variant: {
        ghost: "text-muted hover:text-ink hover:bg-sand",
        accent: "text-accent hover:bg-[var(--accent-weak)]",
      },
      size: {
        sm: "size-7",
        md: "size-9",
      },
    },
    defaultVariants: { variant: "ghost", size: "md" },
  },
);

export interface IconButtonProps
  extends ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof iconButton> {}

const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(
  ({ className, variant, size, ...props }, ref) => (
    <button ref={ref} className={cn(iconButton({ variant, size }), className)} {...props} />
  ),
);
IconButton.displayName = "IconButton";

export default IconButton;
