// config state variables
export const config = $state({
    'timeStart': '2026-11-14T00:00',
    'timeEnd': '2026-11-16T00:00',
    'i': 0,
    'times': [],
})

// colours
export const SPACE_COLORS = [
    "#7C9CFF",  // nebula blue
    "#B28DFF",  // cosmic violet
    "#FF8FD0",  // pulsar pink
    "#FF6B6B",  // mars red
    "#FFA94D",  // solar orange
    "#FFD866",  // starlight gold
    "#7EE0A3",  // aurora green
    "#4FD8D0",  // ion teal
    "#5CC8FF",  // ice-moon cyan
    "#C3CAD9",  // lunar silver
];

export const API_URL = "https://lunar-landing-site-nav.onrender.com/"