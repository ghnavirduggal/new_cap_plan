import type { ReactNode } from "react";
import { GlobalLoaderProvider } from "./_components/GlobalLoader";
import ToastProvider from "./_components/ToastProvider";
import "./globals.css";

export const metadata = {
  title: "CAP-CONNECT",
  description: "Capacity planning",
  // Favicon comes from web/public/assets/icon.png (replace it with the Barclays
  // eagle). Listed here so it wins over any default app-router icon.
  icons: {
    icon: [{ url: "/assets/icon.png", type: "image/png" }],
    shortcut: ["/assets/icon.png"],
    apple: ["/assets/icon.png"]
  }
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body data-cjcrx="addYes">
        <GlobalLoaderProvider>
          <ToastProvider>{children}</ToastProvider>
        </GlobalLoaderProvider>
      </body>
    </html>
  );
}
