<script>
  import SiteCard from "./lib/SiteCard.svelte";
  import SiteSelector from "./lib/SiteSelector.svelte";
  import {sites} from "./shared/sites.svelte.js";
  import {inputs} from "./shared/inputs.svelte.js"
  import TimeSlider from "./lib/TimeSlider.svelte";
  import Moon from "./lib/Moon.svelte";
  import ParameterSelector from "./lib/ParameterSelector.svelte";
    import { config } from "./shared/config.svelte";

  const w = (v) => Number(v) || 0;   // treat null/NaN weights as 0

  const rankedSites = $derived(
      sites
          .map((site, index) => ({
              site,
              index,   // position in the original array
              score:
                  (w(site.scores?.s_science) * w(inputs.science) +
                      w(site.scores?.s_safety) * w(inputs.safety) +
                      w(site.scores?.s_sun) * w(inputs.sun) +
                      w(site.scores?.s_earth) * w(inputs.earth)) /
                  (w(inputs.science) + w(inputs.safety) + w(inputs.sun) + w(inputs.earth)),
          }))
          .sort((a, b) => b.score - a.score)   // highest first
  );


</script>

<div class="app">
    <div class="inputs__wrapper">
        <SiteSelector/>
        <ParameterSelector/>
        <TimeSlider/>
    </div>
    <div class="outputs__wrapper">
        <Moon/>
        <div class="site-cards__wrapper">
            {#each rankedSites as {site, index, score} (site)}
                <SiteCard {site} id={index} {score} />
            {/each}
        </div>

    </div>
</div>

<style>
    .app {
        width: 100vw;
        height: 100vh;
        overflow: hidden;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-sizing: border-box;
    }

    .inputs__wrapper {
        width: 50%;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        align-items: center;
    }

    .outputs__wrapper {
        width: 50%;
        height: 100vh;
        display: flex;
        justify-content: space-between;
        flex-direction: column;
    }

    .site-cards__wrapper {
        width: 100%;
        height: 70vh;
        overflow-x: hidden;
        overflow-y:auto;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        align-items: center;
        background-color: #171717;
        border: 5px solid #272727;
        box-sizing: border-box;
    }
</style>



