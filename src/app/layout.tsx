import type { Metadata } from "next";
import { IBM_Plex_Sans } from "next/font/google";
import "./globals.css";
import { AppErrorBoundary } from "@/components/AppErrorBoundary";

const ibm = IBM_Plex_Sans({
  weight: ["400", "500", "600", "700"],
  subsets: ["latin"],
  variable: "--font-ibm-plex",
});

export const metadata: Metadata = {
  title: "Resume intelligence",
  description: "Parse and compare resumes with Gemini",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={ibm.variable}>
      <body className="font-sans min-h-screen">
        <AppErrorBoundary>{children}</AppErrorBoundary>
      </body>
    </html>
  );
}
