import type React from "react"
import type { Metadata } from "next"
import { Inter } from "next/font/google"
import "./globals.css"
import { Toaster } from "@/app/components/ui/toaster"

const inter = Inter({ subsets: ["latin"] })

export const metadata: Metadata = {
  title: "Smart Tomato Farming",
  description: "AI-powered tomato farming with weather-based scheduling and pest detection",
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="light">
      {/* Added 'light' class to force light mode */}
      <body className={inter.className}>
        {children}
        <Toaster />
      </body>
    </html>
  )
}
