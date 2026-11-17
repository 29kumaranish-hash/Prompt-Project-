import type { Metadata } from 'next';
import './globals.css';
export const metadata: Metadata = { title: 'ResearchCompress AI · A clearer view of research', description: 'Local scientific paper analysis with real LLMLingua-2 compression, source evidence, and measurable token savings.' };
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) { return <html lang="en"><body>{children}</body></html>; }
