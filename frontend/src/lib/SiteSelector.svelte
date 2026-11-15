<script>
    import { LANDING_SITES_PRESETS } from "../shared/landing_sites.js"
    import {addSite} from "../shared/sites.svelte.js";
    import {config} from "../shared/config.svelte.js";
    import {inputs} from "../shared/inputs.svelte.js";

    // lunar presets (planetocentric lat/lng in degrees, east-positive longitude)
    const allPresets = LANDING_SITES_PRESETS.flatMap(g => g.sites);

    // Tte preset matching the current lat/lng, if any
    const selectedPreset = $derived(
        allPresets.find(s =>
            inputs.lat != null && inputs.lng != null &&
            Number(inputs.lat) === s.lat && Number(inputs.lng) === s.lng
        ) ?? null
    );
    const selectedId = $derived(selectedPreset?.id ?? "");

    // name typed by the user. Empty means "use the default".
    let customName = $state("");

    const defaultName = $derived(
        selectedPreset?.name ?? ""
    );

    // update lat lon based on presets
    function onPresetChange(e) {
        const preset = allPresets.find(s => s.id === e.currentTarget.value);
        if (!preset) return;
        inputs.lat = preset.lat;
        inputs.lng = preset.lng;
    }

    function onAddSiteClick() {
        const name = customName.trim() || defaultName;
        addSite(inputs.lat, inputs.lng, name);
        customName = "";   // next site starts from the default again
    }
</script>

<div class="site-selector__wrapper">
    <div class="site-selector__container" style="height: 40vh">
        <h4 class="site-selector__input__header">Lunar Landing Site Dashboard</h4>
        <!-- mission window inputs -->
        <h4 class="site-selector__input__header">Mission Window</h4>
        <div class="site-selector__row">
            <div class="site-selector__input">
                <p class="site-selector__input__label">Start</p>
                <input
                        class="site-selector__input__input"
                        type="datetime-local"
                        id="mission-start"
                        name="Mission Frame Start"
                        bind:value={config.timeStart}
                        min="1970-01-01"
                />
            </div>
            <div class="site-selector__input">
                <p class="site-selector__input__label">End</p>
                <input
                        class="site-selector__input__input"
                        type="datetime-local"
                        id="mission-start"
                        name="Mission Frame Start"
                        bind:value={config.timeEnd}
                        min="1970-01-01"
                />
            </div>
        </div>

        <!-- location inputs -->
        <h4 class="site-selector__input__header">Location</h4>
        <div class="site-selector__row">
            <div class="site-selector__input">
                <p class="site-selector__input__label">Location</p>
                <select
                        name="Known Sites"
                        id="known-sites"
                        class="site-selector__input__input"
                        value={selectedId}
                        onchange={onPresetChange}
                >
                    <option value="">Custom</option>
                    {#each LANDING_SITES_PRESETS as group}
                        <optgroup label={group.label}>
                            {#each group.sites as s}
                                <option value={s.id}>{s.name}</option>
                            {/each}
                        </optgroup>
                    {/each}
                </select>
            </div>
            <div class="site-selector__input">
                <p class="site-selector__input__label">Name</p>
                <input
                        class="site-selector__input__input"
                        type="text"
                        id="site-name"
                        name="Site Name"
                        bind:value={customName}
                        placeholder={defaultName}
                />
            </div>
        </div>
        <div class="site-selector__row">
            <div class="site-selector__input">
                <p class="site-selector__input__label">Latitude (°)</p>
                <input
                        class="site-selector__input__input"
                        type="number"
                        id="lat"
                        name="Site Latitude"
                        bind:value={inputs.lat}
                        min="-90"
                        max="90"
                />
            </div>
            <div class="site-selector__input">
                <p class="site-selector__input__label">Longitude (°)</p>
                <input
                        class="site-selector__input__input"
                        type="number"
                        id="lng"
                        name="Site Longitude"
                        bind:value={inputs.lng}
                        min="-180"
                        max="180"
                />
            </div>
            <button id="add-site" class="site-selector__input__input site-selector__button" type="submit" onclick={onAddSiteClick}>Add</button>
        </div>
    </div>
</div>
