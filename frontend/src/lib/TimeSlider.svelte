<script>
    import {config} from "../shared/config.svelte.js";

    function formatLocalISO(iso) {
        const [datePart, timePart = "00:00"] = iso.split("T");
        const [year, month, day] = datePart.split("-");
        const [hours, minutes] = timePart.split(":");
        return `${year}-${month}-${day} ${hours}:${minutes}`;
    }
</script>

<div class="time__wrapper">
    <input type="range" id="timeId" name="Time" class="time__range"
           min="0"
           max={Math.max(config.times.length - 1, 0)}
           bind:value={config.i} />
    <label for="timeId">{config.times[config.i] ? formatLocalISO(config.times[config.i]) : "–"}</label>
</div>

<style>
    .time__wrapper {
        width: 50vw;
        height: 10vh;
        background-color: #171717;
        border: 5px solid #272727;
        box-sizing: border-box;
        bottom: 0;
        left: 0;
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        z-index: 1000;
    }

    .time__range {
        width: 100%;
        background: #5985E1;
        outline: none;
        opacity: 1;
        height: 5px;
        -webkit-transition: .2s;
        transition: opacity .2s;
    }

    .time__range::-webkit-slider-thumb {
        -webkit-appearance: none;
        appearance: none;
        width: 24px;
        height: 25px;
        border: 0;
        background: url('/clock.svg');
        cursor: pointer;
    }

    .time__range::-moz-range-thumb {
        width: 24px;
        height: 25px;
        border: 0;
        background: url('/clock.svg');
        cursor: pointer;
    }
</style>