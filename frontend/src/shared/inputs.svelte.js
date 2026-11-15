// input state variables
import { LANDER_PRESETS } from "./lander.js";

export const inputs = $state({
    "lat": 26.1,
    "lng": 3.6,
    "safety": 0.25,
    "science": 0.25,
    "sun": 0.25,
    "earth": 0.25,
    "roverDist": 100,
    "return": false,
    "lander": { ...LANDER_PRESETS.find(l => l.id === "standard") },
    "geminiPrompt": ""
})