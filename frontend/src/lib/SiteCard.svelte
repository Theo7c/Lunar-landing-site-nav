<script>
    import {API_URL, SPACE_COLORS} from "../shared/config.svelte.js";
    import sunIcn from "../assets/sun-icn.svg"
    import earthIcn from "../assets/earth-icn.svg"
    import crossIcn from "../assets/cross-icn.svg"
    import resetIcn from "../assets/reset-icn.svg"
    import totalIcn from "../assets/total.svg"
    import safetyIcn from "../assets/safety.svg"
    import scienceIcn from "../assets/science.svg"
    import earthSmallIcn from  "../assets/earth.svg"
    import sunSmallIcn from "../assets/sun.svg"
    import {inputs} from "../shared/inputs.svelte.js";

    import { Line } from 'svelte-chartjs';
    import {
        Chart as ChartJS,
        Title,
        Tooltip,
        Legend,
        LineElement,
        LinearScale,
        PointElement,
        CategoryScale,
        TimeScale,
    } from 'chart.js';
    import 'chartjs-adapter-date-fns';
    import zoomPlugin from 'chartjs-plugin-zoom';
    import { config } from "../shared/config.svelte";
    import { sites } from "../shared/sites.svelte";

    let {site, id, score} = $props();

    // graphs for earth and sun angle
    ChartJS.register(
        Title,
        Tooltip,
        Legend,
        LineElement,
        LinearScale,
        PointElement,
        CategoryScale,
        TimeScale,
        zoomPlugin
    )

    const earthPlotData = $derived({
        datasets: [
            {   // earthAngle Data
                label: "Earth Angle (°)",
                fill: true,
                backgroundColor: '#5985E1',
                borderColor: '#5985E1',
                data: site.data.map(r => ({ x: r.dateTime, y: r.earthEl })),
            },
            {
                label: "Horizon (0°)",
                data: site.data.map(r => ({ x: r.dateTime, y: r.earthHorizonEl })),
                fill: false,
                borderColor: '#888',
                borderWidth: 1,
                borderDash: [6, 4],
                pointRadius: 0,
            },
        ],
    })

    const sunPlotData = $derived({
        datasets: [
            {   // sunAngle Data
                label: "Sun Angle (°)",
                fill: true,
                backgroundColor: '#F19E39',
                borderColor: '#F19E39',
                data: site.data.map(r => ({ x: r.dateTime, y: r.sunEl })),
            },
            {
                label: "Horizon (0°)",
                data: site.data.map(r => ({ x: r.dateTime, y: r.sunHorizonEl })),
                fill: false,
                borderColor: '#888',
                borderWidth: 1,
                borderDash: [6, 4],
                pointRadius: 0,
            },
        ],
    })

    // time scrub
    const verticalLinePlugin = {
        id: 'verticalLine',
        afterDatasetsDraw(chart) {
            const { ctx, chartArea: { top, bottom, left, right }, scales: { x } } = chart;

            const xPx = x.getPixelForValue(new Date(config.times[config.i]).getTime());
            if (xPx < left || xPx > right) return;   // outside visible range

            ctx.save();
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(xPx, top);
            ctx.lineTo(xPx, bottom);
            ctx.stroke();
            ctx.restore();
        },
    };

    // plot configuration
    const plotOptions = $derived({
        elements: {
            point: {
                radius: 0
            }
        },
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: false,
            },
            zoom: {
                limits: {
                    x: {
                        max: 'original'
                    }
                },
                pan: {
                    enabled: true,
                    mode: 'x',
                },
                zoom: {
                    wheel: {
                        enabled: true,
                    },
                    pinch: {
                        enabled: true
                    },
                    mode: 'x',
                }
            }
        },
        scales: {
            x: {
                type: "time",
                min: config.timeStart,
                max: config.timeEnd,
                time: {
                    // no `unit`: let Chart.js choose based on the range
                    tooltipFormat: "yyyy-MM-dd HH:mm",
                    displayFormats: {
                        minute: "HH:mm",
                        hour: "HH:mm",
                        day: "MMM d",
                    },
                },
                ticks: {
                    maxRotation: 0,
                    autoSkip: true,
                    maxTicksLimit: 6,
                    font: {size: 10},
                    color: '#999',
                },
                grid: {color: 'rgba(255,255,255,0.06)'},
            },
            y: {
                ticks: {
                    maxTicksLimit: 5,
                    font: {size: 10},
                    color: '#999',
                    callback: (v) => `${v}°`,
                },
                grid: {color: 'rgba(255,255,255,0.06)'},
                title: {display: false},
            },
        }
    })

    let earthChart = $state(null);
    let sunChart = $state(null);

    // refresh plot on change of time
    $effect(() => {
        config.i;
        earthChart?.draw();
        sunChart?.draw();
    });

    // format strings from api to correct decimal places
    const fmt = (v, d) => {
        const n = Number(v);
        return v == null || Number.isNaN(n) ? "–" : n.toFixed(d);
    };

    // SSI image
    const routeMapBase = API_URL + "api/site/route-map";

    const routeMapUrl = $derived(
        `${routeMapBase}?` + new URLSearchParams({
            lat: site.lat,
            lng: site.lng,
            r_max_km: inputs.roverDist,
            return_to_lander: !!inputs.return,
            area_radius_m: inputs.lander.area_radius_m,
            dpi: 100,   // card-sized, the default 200 is slower and heavier
        })
    );

    $effect(() => {
        // update border colour based on order in list
        document.getElementsByClassName("site-card__container")[id].style.borderColor = SPACE_COLORS[id]
    });

</script>

<div class="site-card__wrapper">
    <div class="site-card__container">
        <div class="site-card__scores">
            <div class="site-card__score">
                <img src={totalIcn}/>
                {score.toFixed(2)}
            </div>
            <div class="site-card__score">
                <img src={scienceIcn}/>
                {site.scores.s_science.toFixed(2)}
            </div>
            <div class="site-card__score">
                <img src={safetyIcn}/>
                {site.scores.s_safety.toFixed(2)}
            </div>
            <div class="site-card__score">
                <img src={sunSmallIcn}/>
                {site.scores.s_sun.toFixed(2)}
            </div>
            <div class="site-card__score">
                <img src={earthSmallIcn}/>
                {site.scores.s_earth.toFixed(2)}
            </div>
        </div>

        <div class="site-card__actions">
            <img class="site-card__action" src={resetIcn} alt="reset zoom" onclick={() => {earthChart.resetZoom(); sunChart.resetZoom()}}/>
            <img class="site-card__action" src={crossIcn} alt="remove site" onclick={() => sites.splice(id, 1)}/>
        </div>
        <div class="site-card__header-row">
            <div class="site-card__header-col">
                <p class="site-card__header-col__title site-card__header-coords">
                    {site.name} · {fmt(site.lat, 2)}° {fmt(site.lng, 2)}° {fmt(site.alt, 1)} km
                </p>
            </div>
            <div class="site-card__header-col">
                <img class="site-card__header-col__icn" src={sunIcn} alt="sun icon"/>
                <h4 class="site-card__header-col__title">{(site.data[config.i].sunEl.toFixed(2))}°</h4>
            </div>
            <div class="site-card__header-col">
                <img class="site-card__header-col__icn" src={earthIcn} alt="earth icon"/>
                <h4 class="site-card__header-col__title">{site.data[config.i].earthEl.toFixed(2)}°</h4>
            </div>
        </div>
        <div class="site-card__outputs">
            <div class="site-card__plots">
                <div class="site-card__plot">
                    <Line bind:chart={sunChart} data={sunPlotData} options={plotOptions}
                          plugins={[verticalLinePlugin]} />
                </div>
                <div class="site-card__plot">
                    <Line bind:chart={earthChart} data={earthPlotData} options={plotOptions}
                          plugins={[verticalLinePlugin]} />
                </div>
            </div>
            <div class="site-card__map">
                <img src={routeMapUrl} alt="Loading rover route map..." class="site-card__map"/>
            </div>
        </div>


    </div>
</div>

<style>
    .site-card__container {
        position: relative; /* anchor for the absolute icons */
        width: 100%;
        background-color: #171717;
        border: 5px solid #272727;
        box-sizing: border-box;
        border-radius: 2.5px;
    }

    .site-card__scores {
        display: flex;
        align-items: center;
        justify-content: flex-start;
    }

    .site-card__score {
        padding: 0 .5em 0 .5em;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    .site-card__actions {
        position: absolute;
        top: 4px;
        right: 0px;
        display: flex;
        gap: 4px;
        z-index: 2;
    }

    .site-card__wrapper {
        margin-bottom: .5em;
        width: 100%;
    }

    .site-card__header-row {
        display: flex;
        justify-content: space-evenly;
        align-items: center;
        height: 30px
    }

    .site-card__header-col {
        background-color: #212123;
        width: 100%;
        height: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    .site-card__header-coords {
        font-size: 15px;
    }

    .site-card__header-col__icn {
        padding: .25em;
        height: 25px;
        width: auto;
    }

    .site-card__outputs {
        display: flex;
        align-items: stretch;
        gap: .5em;
        width: 100%;
    }

    .site-card__plots {
        flex: 1 1 0;
        min-width: 0;
    }

    .site-card__plot {
        height: 75px;
        width: 100%
    }

    .site-card__map {
        flex: 0 0 150px;
        width: 150px;
    }

    .site-card__map img {
        display: block;
        height: 150px;
        width: 150px;
    }
</style>