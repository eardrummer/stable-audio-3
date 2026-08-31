import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import LiveApp from './LiveApp.tsx'

createRoot(document.getElementById('live-root')!).render(
  <StrictMode>
    <LiveApp />
  </StrictMode>,
)
