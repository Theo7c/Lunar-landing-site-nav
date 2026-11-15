import { untrack } from "svelte";
import { API_URL, config } from './config.svelte.js'
import {inputs} from "./inputs.svelte.js";

export const sites = $state([])

// date conversion logic
const MONTHS = { Jan:"01", Feb:"02", Mar:"03", Apr:"04", May:"05", Jun:"06",
    Jul:"07", Aug:"08", Sep:"09", Oct:"10", Nov:"11", Dec:"12" };

const toLocalISO = s => {
    const [d, t] = s.split(" ");
    const [y, mon, day] = d.split("-");
    return `${y}-${MONTHS[mon]}-${day}T${t}`;
};

const RESOLUTION_MS = {
    minute: 60_000,
    hour:   3_600_000,
    day:    86_400_000,
    month:  86_400_000,
    year:   86_400_000,
};

// update time frames
function applyTimes(data, resolution) {
    config.times = data.map(r => r.dateTime);
    config.step = RESOLUTION_MS[resolution] ?? config.step;
    config.i = Math.min(config.i, Math.max(config.times.length - 1, 0));
}

async function fetchSite(lat, lng, start, end, signal) {
    // query API for data for a specific site
    const params = new URLSearchParams({
        lat,
        lng,
        time_start: start,
        time_end: end,
    });

    const res = await fetch(`${API_URL}?${params}`, { signal });
    if (!res.ok) throw new Error(`API error ${res.status}`);

    const raw = await res.json();
    raw.data.forEach(r => { r.dateTime = toLocalISO(r.dateTime); });

    return {
        lat,
        lng,
        alt: raw.alt,
        data: raw.data,
        resolution: raw.resolution,
        scores: raw.scores,
    };
}

export async function addSite(lat, lng, name) {
    // add site to sites
    try {
        const { resolution, ...site } = await fetchSite(lat, lng, config.timeStart, config.timeEnd);
        sites.push({ ...site, 'name': name });
        if (sites.length === 1) applyTimes(site.data, resolution);
    } catch (e) {
        console.error("Failed to add site", e);
    }
}

// re-fetch data on time period change
$effect.root(() => {
    $effect(() => {
        const start = config.timeStart;
        const end = config.timeEnd;

        const controller = new AbortController(); //  ensures no overlaps

        (async () => {
            try {
                const targets = untrack(() => sites.map(s => ({ lat: s.lat, lng: s.lng })));
                if (targets.length === 0) return;

                const results = await Promise.all(
                    targets.map(t => fetchSite(t.lat, t.lng, start, end, controller.signal))
                );

                untrack(() => {
                    results.forEach((res, i) => {
                        if (sites[i]) sites[i].data = res.data;
                    });
                    applyTimes(results[0].data, results[0].resolution);
                });
            } catch (e) {
                if (e.name !== "AbortError") console.error(e);
            }
        })();

        return () => controller.abort();
    });
});