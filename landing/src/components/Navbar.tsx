import { motion } from 'framer-motion'
import { links } from '../site'

export default function Navbar() {
  return (
    <motion.nav
      className="nav-bar"
      initial={{ opacity: 0, y: -16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, ease: 'easeOut' }}
      style={{ position: 'fixed', top: 0, left: 0, right: 0, zIndex: 50, display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '26px 44px' }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '44px' }}>
        <a href="/" aria-label="Dubimator" style={{ display: 'flex', alignItems: 'center', gap: '10px', textDecoration: 'none' }}>
          <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" xmlns="http://www.w3.org/2000/svg">
            <path d="M6 22V4a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v18Z" />
            <path d="M6 12H4a2 2 0 0 0-2 2v6a2 2 0 0 0 2 2h2" />
            <path d="M18 9h2a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2h-2" />
            <path d="M10 6h4" />
            <path d="M10 10h4" />
            <path d="M10 14h4" />
            <path d="M10 18h4" />
          </svg>
          <span style={{ fontSize: '17px', fontWeight: 600, fontFamily: "'Plus Jakarta Sans', sans-serif", letterSpacing: '-0.01em', color: '#fff' }}>Dubimator</span>
        </a>
        <div className="nav-links" style={{ display: 'flex', alignItems: 'center', gap: '30px' }}>
          {links.nav.map((link) => (
            <a
              key={link.label}
              href={link.href}
              style={{ fontSize: '14px', fontWeight: 500, fontFamily: "'Inter', sans-serif", color: 'rgba(255,255,255,0.82)', textDecoration: 'none' }}
            >
              {link.label}
            </a>
          ))}
        </div>
      </div>
      <motion.a
        href={links.architecture}
        whileHover={{ scale: 1.04 }}
        whileTap={{ scale: 0.97 }}
        style={{
          padding: '11px 24px',
          borderRadius: '999px',
          fontSize: '14px',
          fontWeight: 600,
          fontFamily: "'Inter', sans-serif",
          color: '#111',
          textDecoration: 'none',
          background: '#fff',
          boxShadow: '0 4px 18px rgba(0,0,0,0.25)',
        }}
      >
        Architecture
      </motion.a>
    </motion.nav>
  )
}
