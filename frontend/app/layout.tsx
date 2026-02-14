import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Dance Video Generator - AI-Powered Dance Videos',
  description: 'Create Sora/Kling-quality dance videos with facial expressions and text prompts',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
