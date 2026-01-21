import type { ReactNode } from "react";
import { GlobalLoaderProvider } from "./_components/GlobalLoader";
import ToastProvider from "./_components/ToastProvider";
import "./globals.css";

export const metadata = {
  title: "CAP-CONNECT",
  description: "Capacity planning"
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
