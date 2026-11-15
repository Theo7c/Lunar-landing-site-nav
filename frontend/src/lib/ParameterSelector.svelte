<script>
    import { LANDER_PRESETS } from "../shared/lander.js";
    import { addSite } from "../shared/sites.svelte.js";
    import { inputs } from "../shared/inputs.svelte.js";
    import { API_URL } from "../shared/config.svelte.js";
    import { config } from "../shared/config.svelte.js";

    // update lander parameters based on selection
    function onLanderChange(e) {
        const preset = LANDER_PRESETS.find(l => l.id === e.currentTarget.value);
        if (preset) inputs.lander = { ...preset };   // copy so the preset constants are never mutated
    }

    // adds best 5 sites for mission parameters
    async function onAllSiteSearch() {

        const params = new URLSearchParams({
            "w_safety": inputs.safety,
            "w_science": inputs.science,
            "w_sun": inputs.sun,
            "w_earth": inputs.earth,
            "time_start": config.timeStart,
            "time_end": config.timeEnd,
            "r_max_km": inputs.roverDist,
        });

        const res = await fetch(`${API_URL}api/sites/scores?${params}`);
        if (!res.ok) throw new Error(`API error ${res.status}`);

        const raw = await res.json();

        for (let i = 0; i < raw.top_sites.length; i++) {
            await addSite(raw.top_sites[i].lat, raw.top_sites[i].lon); // TODO: implement more efficiency
        }

    }

    // gemini weight helper
    const justification = $state({
        "msg": ""
    })

    // ask gemini
    async function onAsk() {
        const prompt = inputs.geminiPrompt;
        // query API for data for a specific site
        const params = new URLSearchParams({
            "query": prompt
        });

        const res = await fetch(`${API_URL}api/ai?${params}`);
        if (!res.ok) throw new Error(`API error ${res.status}`);

        const raw = await res.json();
        inputs.safety = raw.weights[0];
        inputs.science = raw.weights[1];
        inputs.sun = raw.weights[2];
        inputs.earth = raw.weights[3];

        // update justification
        justification["msg"] = raw.justifications

        // update with suggested latitude and longitude
        inputs.lat = raw.coords[0];
        inputs.lng = raw.coords[1];
    }
</script>

<div class="site-selector__wrapper">
    <div class="site-selector__container" style="height: 50vh">

        <!-- capabilities -->
        <h4 class="site-selector__input__header">Capabilities</h4>
        <div class="site-selector__row">
            <div class="site-selector__input">
                <p class="site-selector__input__label">Rover Distance (km)</p>
                <input
                        class="site-selector__input__input"
                        type="number"
                        id="roverdist"
                        name="Rover Distance (km)"
                        bind:value={inputs.roverDist}
                        min="1"
                        max="50"
                        step="1"
                />
            </div>
            <div class="site-selector__input">
                <p class="site-selector__input__label">Return</p>
                <input
                        class="site-selector__input__input"
                        type="checkbox"
                        id="return"
                        name="Return"
                        bind:checked={inputs.return}
                />
            </div>
            <div class="site-selector__input">
                <p class="site-selector__input__label">Lander</p>
                <select
                        name="Lander"
                        id="lander"
                        class="site-selector__input__input"
                        value={inputs.lander?.id}
                        onchange={onLanderChange}
                >
                    {#each LANDER_PRESETS as l}
                        <option value={l.id}>{l.label}</option>
                    {/each}
                </select>
            </div>
        </div>

        <!-- weights -->
        <h4 class="site-selector__input__header">Weights</h4>
        <div class="site-selector__row">
            <div class="site-selector__input">
                <p class="site-selector__input__label">Safety</p>
                <input
                        class="site-selector__input__input"
                        type="range"
                        id="safety"
                        name="Safety"
                        bind:value={inputs.safety}
                        min="0"
                        max="1"
                        step="0.05"
                />
            </div>
            <div class="site-selector__input">
                <p class="site-selector__input__label">Solar Power</p>
                <input
                        class="site-selector__input__input"
                        type="range"
                        id="sun"
                        name="Solar Power"
                        bind:value={inputs.sun}
                        min="0"
                        max="1"
                        step="0.05"
                />
            </div>
        </div>
        <div class="site-selector__row">
            <div class="site-selector__input">
                <p class="site-selector__input__label">Direct Earth Contact</p>
                <input
                        class="site-selector__input__input"
                        type="range"
                        id="earth"
                        name="Direct Earth Contact"
                        bind:value={inputs.earth}
                        min="0"
                        max="1"
                        step="0.05"
                />
            </div>
            <div class="site-selector__input">
                <p class="site-selector__input__label">Scientific Interest</p>
                <input
                        class="site-selector__input__input"
                        type="range"
                        id="science"
                        name="Scientific Interest"
                        bind:value={inputs.science}
                        min="0"
                        max="1"
                        step="0.05"
                />
            </div>
        </div>
        <button id="add-site" class="site-selector__input__input site-selector__button" type="submit" onclick={onAllSiteSearch}>Search All Sites</button>
        <!-- gemini ask inputs -->
        <div class="site-selector__row">
            <div class="site-selector__input">
                <input
                        class="site-selector__input__input"
                        style="width: 40vw"
                        type="text"
                        id="geminiPrompt"
                        name="Gemini Prompt"
                        bind:value={inputs.geminiPrompt}
                        placeholder="Input mission information to generate weights."
                />
                <button id="add-site" class="site-selector__input__input site-selector__button" type="submit" onclick={onAsk}>Ask</button>
            </div>
        </div>
        <p class="site-selector__justification">{justification["msg"]}</p>
    </div>
</div>
