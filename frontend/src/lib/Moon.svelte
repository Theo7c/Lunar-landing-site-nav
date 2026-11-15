<script>
    import { onMount, onDestroy } from "svelte";
    import * as THREE from "three";
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

    import { inputs } from "../shared/inputs.svelte.js";
    import { sites } from "../shared/sites.svelte";
    import { SPACE_COLORS } from "../shared/config.svelte.js";

    // initialise THREE.js
    let canvas;
    let renderer, camera, animId, resizeObserver;

    // define moon scale
    const R_MOON_KM = 1737.4;
    const MIN_KM = -9.1;
    const MAX_KM = 10.8;
    const EXAGGERATION = 10; // 1 = true scale

    const displacementScale = ((MAX_KM - MIN_KM) / R_MOON_KM) * EXAGGERATION;
    const displacementBias  = (MIN_KM / R_MOON_KM) * EXAGGERATION;

    let markerGroup = $state.raw(null);

    // helper functions
    function latLonToVector3(lat, lon, radius) {
        const phi = THREE.MathUtils.degToRad(90 - lat);
        const theta = THREE.MathUtils.degToRad(lon + 90);
        return new THREE.Vector3().setFromSphericalCoords(radius, phi, theta);
    }

    onMount(() => {
        // create scene, renderer and moon
        const scene = new THREE.Scene();
        camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
        camera.position.x = -2;
        scene.add(camera)

        renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

        // define moon geometry
        const geometry = new THREE.SphereGeometry(1, 1024, 512);

        const loader = new THREE.TextureLoader();
        const colourMap = loader.load("lroc_color_poles_1k.jpg")
        colourMap.colorSpace = THREE.SRGBColorSpace;

        const depthMap = loader.load("ldem_3_8bit.jpg")

        const material = new THREE.MeshStandardMaterial({
            map: colourMap,
            displacementMap: depthMap,
            displacementScale,
            displacementBias,
            bumpMap: depthMap,
            bumpScale: 0.5,
        });

        const mesh = new THREE.Mesh(geometry, material);
        scene.add(mesh);

        // lighting
        const ambient = new THREE.AmbientLight(0xffffff, 0.1);
        scene.add(ambient);

        const sun = new THREE.DirectionalLight(0xffffff, 1.5);
        sun.position.set(-2, 1, -2);
        scene.add(sun);

        // markers
        const group = new THREE.Group();
        scene.add(group);
        markerGroup = group;

        // orbit controls
        const controls = new OrbitControls( camera, renderer.domElement );
        camera.lookAt(0, 0, 0);
        controls.enableDamping = true;
        controls.update();

        // resizing
        const resize = () => {
            const w = canvas.clientWidth;
            const h = canvas.clientHeight;
            renderer.setSize(w, h, false);
            camera.aspect = w / h;
            camera.updateProjectionMatrix();
        };

        // ray casting logic
        const raycaster = new THREE.Raycaster();
        const pointer = new THREE.Vector2();
        let downX = 0, downY = 0;

        const onDown = (e) => { downX = e.clientX; downY = e.clientY; };

        const onUp = (e) => {
            if (Math.hypot(e.clientX - downX, e.clientY - downY) > 4) return; // it was a drag

            const rect = canvas.getBoundingClientRect();
            pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
            pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

            raycaster.setFromCamera(pointer, camera);
            const hit = raycaster.intersectObject(mesh)[0];
            if (!hit) return;

            // SphereGeometry UVs: u=0.5 is 0° longitude, v=0.5 is the equator
            const lng = (hit.uv.x - 0.5) * 360;
            const lat = (hit.uv.y - 0.5) * 180;
            inputs.lat = lat
            inputs.lng = lng
            console.log(`lat ${lat.toFixed(2)}°, lon ${lng.toFixed(2)}°`, hit.point);
        };

        canvas.addEventListener("mousedown", onDown);
        canvas.addEventListener("mouseup", onUp);

        // animation logic
        resizeObserver = new ResizeObserver(resize);
        resizeObserver.observe(canvas);
        resize();

        const t0 = Date.now()
        const omega = 2*Math.PI/(100 * 1000)

        const animate = () => {
            animId = requestAnimationFrame(animate);

            const a = omega * (Date.now() - t0);
            sun.position.set(-2 * Math.sin(a), Math.cos(a), 0);

            controls.update();
            renderer.render(scene, camera);
        };
        animate();
    });

    onDestroy(() => {
        cancelAnimationFrame(animId);
        resizeObserver?.disconnect();
        renderer?.dispose();
    });

    // draw sites on site recalculation
    const markerGeometry = new THREE.SphereGeometry(0.02, 16, 8);

    function syncMarkers(list) {
        if (!markerGroup) return;
        markerGroup.clear();

        for (let i = 0; i < list.length; i++) {
            const site = list[i];
            const r = 1 + (Number(site.alt) * EXAGGERATION) / R_MOON_KM;
            const markerMaterial = new THREE.MeshBasicMaterial({ color: SPACE_COLORS[i]});
            const marker = new THREE.Mesh(markerGeometry, markerMaterial);
            marker.position.copy(latLonToVector3(Number(site.lat), Number(site.lng), r));
            markerGroup.add(marker);
        }
    }

    $effect(() => {
        syncMarkers(sites);
    });

</script>

<canvas bind:this={canvas} class="moon__canvas"></canvas>

<style>
    .moon__canvas {
        width: 50vw;
        height: 30vh;
    }
</style>