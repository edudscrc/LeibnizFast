import { defineConfig } from 'vitepress';

export default defineConfig({
  title: 'LeibnizFast',
  description: 'GPU-accelerated 2D matrix visualization for the browser',

  themeConfig: {
    nav: [
      { text: 'Guide', link: '/getting-started' },
      { text: 'API Reference', link: '/api/leibniz-fast' },
    ],

    sidebar: [
      {
        text: 'Introduction',
        items: [{ text: 'Getting Started', link: '/getting-started' }],
      },
      {
        text: 'Guide',
        items: [
          { text: 'Initialization', link: '/guide/initialization' },
          { text: 'Static Data', link: '/guide/static-data' },
          { text: 'Chart Customization', link: '/guide/chart-customization' },
          { text: 'Mouse Interaction', link: '/guide/interaction' },
          {
            text: 'Streaming: Full Frame',
            link: '/guide/streaming-full-frame',
          },
          {
            text: 'Streaming: Waterfall',
            link: '/guide/streaming-waterfall',
          },
        ],
      },
      {
        text: 'API Reference',
        items: [
          { text: 'LeibnizFast', link: '/api/leibniz-fast' },
          { text: 'Types', link: '/api/types' },
          { text: 'Axes Utilities', link: '/api/axes' },
        ],
      },
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/edudscrc/LeibnizFast' },
    ],

    search: {
      provider: 'local',
    },

    footer: {
      message: 'Released under the MIT License.',
    },
  },

  markdown: {
    lineNumbers: true,
  },
});
