import React from 'react'
import './globals.css'

export default function RootLayout({ children }) {
  return (
    <html>
      <head>
        <title>ClarityAI | Premium Video Enhancement</title>
      </head>
      <body>
        {children}
      </body>
    </html>
  )
}
