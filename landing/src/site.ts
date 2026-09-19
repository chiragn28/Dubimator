// Every outbound link on the page. The demo is the Streamlit app (docker compose service
// `demo`); set VITE_DEMO_URL at build time to point at a public tunnel instead of localhost.
const demo = (import.meta.env.VITE_DEMO_URL ?? 'http://localhost:8501').replace(/\/$/, '')

export const links = {
  demo,
  architecture: 'https://dubimator-architecture.vercel.app',
  nav: [
    { label: 'Valuation', href: `${demo}/Price_and_forecast` },
    { label: 'Search', href: `${demo}/Search` },
    { label: 'Listing Check', href: `${demo}/Listing_check` },
    { label: 'Areas', href: `${demo}/Area_explorer` },
  ],
  // Leave a link empty to hide its icon.
  github: 'https://github.com/chiragn28/Dubimator',
  linkedin: '',
}
