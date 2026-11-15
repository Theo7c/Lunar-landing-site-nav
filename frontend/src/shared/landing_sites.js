// lunar landing site presets (planetocentric lat/lng in degrees, east-positive longitude)
export const LANDING_SITES_PRESETS = [
    {
        label: "Apollo landing sites",
        sites: [
            {id: "apollo-11", name: "Apollo 11",  lat:   0.6742, lng:  23.4731},
            {id: "apollo-12", name: "Apollo 12",  lat:  -3.0128, lng: -23.4219},
            {id: "apollo-14", name: "Apollo 14",  lat:  -3.6453, lng: -17.4714},
            {id: "apollo-15", name: "Apollo 15",  lat:  26.1322, lng:   3.6334},
            {id: "apollo-16", name: "Apollo 16",  lat:  -8.9734, lng:  15.5011},
            {id: "apollo-17", name: "Apollo 17",  lat:  20.1908, lng:  30.7717},
        ],
    },
    {
        label: "Robotic landers",
        sites: [
            {id: "change-3",      name: "Chang'e 3",      lat:  44.1214, lng:  -19.5116},
            {id: "change-4",      name: "Chang'e 4",      lat: -45.4561, lng:  177.5885},
            {id: "change-5",      name: "Chang'e 5",      lat:  43.0576, lng:  -51.9161},
            {id: "chandrayaan-3", name: "Chandrayaan-3",  lat: -69.3733, lng:   32.3191},
        ],
    },
    {
        label: "Polar / interesting terrain",
        sites: [
            {id: "shackleton", name: "Shackleton Crater",       lat: -89.54, lng: 129.78},
            {id: "malapert",   name: "Mons Malapert",           lat: -85.99, lng:   0.12},
            {id: "peary",      name: "Peary Crater (north)",    lat:  88.63, lng:  33.00},
        ],
    },
];
