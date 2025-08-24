/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./pages/**/*.{js,jsx,ts,tsx}",      // Pages Router
    "./components/**/*.{js,jsx,ts,tsx}",
    "./app/**/*.{js,jsx,ts,tsx}",        // (si tuvieras App Router)
  ],
  theme: {
    extend: {},
  },
  plugins: [],
};
