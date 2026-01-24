const sharp = require('sharp');
const path = require('path');
const fs = require('fs');

// RoboTrader icon - geometric robot head with chart element
const createIconSVG = (size) => `
<svg width="${size}" height="${size}" viewBox="0 0 1024 1024" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#0c0c0e"/>
      <stop offset="100%" style="stop-color:#08080a"/>
    </linearGradient>
    <linearGradient id="robotGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#34d399"/>
      <stop offset="100%" style="stop-color:#10b981"/>
    </linearGradient>
    <linearGradient id="chartGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#3b82f6"/>
      <stop offset="100%" style="stop-color:#10b981"/>
    </linearGradient>
    <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="8" result="blur"/>
      <feMerge>
        <feMergeNode in="blur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>

  <!-- Background -->
  <rect width="1024" height="1024" fill="url(#bgGrad)"/>

  <!-- Outer glow ring -->
  <circle cx="512" cy="512" r="380" fill="none" stroke="url(#robotGrad)" stroke-width="2" opacity="0.3"/>

  <!-- Robot head outline -->
  <rect x="280" y="260" width="464" height="400" rx="60" fill="none" stroke="url(#robotGrad)" stroke-width="8" filter="url(#glow)"/>

  <!-- Antenna -->
  <line x1="512" y1="180" x2="512" y2="260" stroke="url(#robotGrad)" stroke-width="8" stroke-linecap="round"/>
  <circle cx="512" cy="168" r="24" fill="url(#robotGrad)"/>

  <!-- Eyes -->
  <rect x="340" y="360" width="120" height="80" rx="16" fill="url(#robotGrad)"/>
  <rect x="564" y="360" width="120" height="80" rx="16" fill="url(#robotGrad)"/>

  <!-- Eye highlights -->
  <rect x="356" y="376" width="40" height="32" rx="8" fill="#08080a" opacity="0.6"/>
  <rect x="580" y="376" width="40" height="32" rx="8" fill="#08080a" opacity="0.6"/>

  <!-- Chart line (mouth area) - representing trading -->
  <polyline
    points="320,560 400,540 460,580 520,500 580,550 640,480 704,520"
    fill="none"
    stroke="url(#chartGrad)"
    stroke-width="12"
    stroke-linecap="round"
    stroke-linejoin="round"
    filter="url(#glow)"
  />

  <!-- Grid dots (subtle) -->
  <circle cx="320" cy="720" r="4" fill="#10b981" opacity="0.3"/>
  <circle cx="400" cy="720" r="4" fill="#10b981" opacity="0.3"/>
  <circle cx="480" cy="720" r="4" fill="#10b981" opacity="0.3"/>
  <circle cx="560" cy="720" r="4" fill="#10b981" opacity="0.3"/>
  <circle cx="640" cy="720" r="4" fill="#10b981" opacity="0.3"/>
  <circle cx="720" cy="720" r="4" fill="#10b981" opacity="0.3"/>
</svg>
`;

// Splash icon - simpler, larger elements
const createSplashSVG = () => `
<svg width="1284" height="2778" viewBox="0 0 1284 2778" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#0c0c0e"/>
      <stop offset="100%" style="stop-color:#08080a"/>
    </linearGradient>
    <linearGradient id="robotGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#34d399"/>
      <stop offset="100%" style="stop-color:#10b981"/>
    </linearGradient>
  </defs>

  <!-- Background -->
  <rect width="1284" height="2778" fill="url(#bgGrad)"/>

  <!-- Centered robot icon -->
  <g transform="translate(392, 1139)">
    <!-- Robot head outline -->
    <rect x="50" y="80" width="400" height="340" rx="50" fill="none" stroke="url(#robotGrad)" stroke-width="8"/>

    <!-- Antenna -->
    <line x1="250" y1="20" x2="250" y2="80" stroke="url(#robotGrad)" stroke-width="8" stroke-linecap="round"/>
    <circle cx="250" cy="10" r="20" fill="url(#robotGrad)"/>

    <!-- Eyes -->
    <rect x="110" y="160" width="100" height="70" rx="14" fill="url(#robotGrad)"/>
    <rect x="290" y="160" width="100" height="70" rx="14" fill="url(#robotGrad)"/>

    <!-- Chart line -->
    <polyline
      points="80,320 150,300 200,340 250,270 300,310 350,250 420,290"
      fill="none"
      stroke="url(#robotGrad)"
      stroke-width="10"
      stroke-linecap="round"
      stroke-linejoin="round"
    />
  </g>
</svg>
`;

async function generateIcons() {
  const assetsDir = path.join(__dirname, '..', 'assets', 'images');

  console.log('Generating app icons...');

  // Main icon (1024x1024)
  await sharp(Buffer.from(createIconSVG(1024)))
    .png()
    .toFile(path.join(assetsDir, 'icon.png'));
  console.log('✓ icon.png (1024x1024)');

  // Adaptive icon (1024x1024)
  await sharp(Buffer.from(createIconSVG(1024)))
    .png()
    .toFile(path.join(assetsDir, 'adaptive-icon.png'));
  console.log('✓ adaptive-icon.png (1024x1024)');

  // Favicon (48x48)
  await sharp(Buffer.from(createIconSVG(1024)))
    .resize(48, 48)
    .png()
    .toFile(path.join(assetsDir, 'favicon.png'));
  console.log('✓ favicon.png (48x48)');

  // Splash icon
  await sharp(Buffer.from(createSplashSVG()))
    .png()
    .toFile(path.join(assetsDir, 'splash-icon.png'));
  console.log('✓ splash-icon.png (1284x2778)');

  console.log('\nDone! Icons saved to assets/images/');
}

generateIcons().catch(console.error);
