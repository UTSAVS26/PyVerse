module.exports = {
    content: ["./src/**/*.{js,jsx,ts,tsx}"],
    theme: {
        extend: {
            fontFamily: {
                Opensans: "'Open Sans', sans-serif",
            },
        },
    },
    plugins: [require("@tailwindcss/forms")],
};
