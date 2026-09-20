const API = 'https://api.open-meteo.com/v1/forecast';
const GEOCODER = 'https://geocoding-api.open-meteo.com/v1/search';
const DEFAULT_PLACE = { name: 'Lausanne', latitude: 46.5197, longitude: 6.6323, country: 'Switzerland', admin1: 'Vaud' };
const weatherCodes = {
  0: ['Clear sky', '☀️'], 1: ['Mainly clear', '🌤️'], 2: ['Partly cloudy', '⛅'], 3: ['Overcast', '☁️'],
  45: ['Foggy', '🌫️'], 48: ['Rime fog', '🌫️'], 51: ['Light drizzle', '🌦️'], 53: ['Drizzle', '🌦️'],
  55: ['Heavy drizzle', '🌧️'], 61: ['Light rain', '🌦️'], 63: ['Rain', '🌧️'], 65: ['Heavy rain', '🌧️'],
  71: ['Light snow', '🌨️'], 73: ['Snow', '🌨️'], 75: ['Heavy snow', '❄️'], 80: ['Rain showers', '🌦️'],
  81: ['Rain showers', '🌧️'], 82: ['Heavy showers', '⛈️'], 95: ['Thunderstorms', '⛈️'],
  96: ['Thunderstorms', '⛈️'], 99: ['Thunderstorms', '⛈️'],
};
const $ = (id) => document.getElementById(id);
let currentPlace = DEFAULT_PLACE;
let map = { zoom: 7, centerX: 0, centerY: 0, dragging: false, startX: 0, startY: 0, baseX: 0, baseY: 0, radarPath: null, radarTime: null };

init();

function init() {
  $('search-form').addEventListener('submit', (event) => {
    event.preventDefault();
    const query = $('search-input').value.trim();
    if (query) loadPlace(query);
  });
  $('locate').addEventListener('click', locate);
  $('zoom-in').addEventListener('click', () => changeZoom(1));
  $('zoom-out').addEventListener('click', () => changeZoom(-1));
  const mapEl = $('map');
  mapEl.addEventListener('pointerdown', startPan);
  mapEl.addEventListener('pointermove', pan);
  mapEl.addEventListener('pointerup', endPan);
  mapEl.addEventListener('pointercancel', endPan);
  loadForecast(DEFAULT_PLACE);
}

async function loadPlace(query) {
  setLoading();
  try {
    const response = await fetch(`${GEOCODER}?name=${encodeURIComponent(query)}&count=1&language=en&format=json`);
    if (!response.ok) throw new Error('The location service is unavailable.');
    const result = await response.json();
    if (!result.results?.length) throw new Error(`No locations found for “${query}”.`);
    await loadForecast(result.results[0]);
  } catch (error) {
    showError(error.message);
  }
}

function locate() {
  if (!navigator.geolocation) return showError('Location services are not supported by this browser.');
  setLoading();
  navigator.geolocation.getCurrentPosition(
    ({ coords }) => loadForecast({ name: 'Current location', latitude: coords.latitude, longitude: coords.longitude }).catch((error) => showError(error.message)),
    () => showError('We could not access your location. Search for a city instead.'),
    { enableHighAccuracy: false, timeout: 8000 },
  );
}

async function loadForecast(place) {
  const params = new URLSearchParams({
    latitude: place.latitude, longitude: place.longitude, timezone: 'auto', forecast_days: 7,
    current: 'temperature_2m,relative_humidity_2m,apparent_temperature,is_day,precipitation,weather_code,wind_speed_10m',
    hourly: 'temperature_2m,precipitation_probability,weather_code,wind_speed_10m',
    daily: 'weather_code,temperature_2m_max,temperature_2m_min,sunrise,sunset,precipitation_probability_max',
  });
  try {
    const response = await fetch(`${API}?${params}`);
    if (!response.ok) throw new Error('The forecast service returned an error.');
    const forecast = await response.json();
    currentPlace = place;
    render(forecast);
    await renderMap(place);
  } catch (error) {
    showError(`Could not load the forecast. ${error.message}`);
  }
}

function render(data) {
  $('weather').hidden = true;
  $('dashboard').hidden = false;
  $('notice').classList.remove('visible');
  const current = data.current;
  const [label, icon] = describe(current.weather_code);
  $('place-name').textContent = currentPlace.name;
  $('place-region').textContent = [currentPlace.admin1, currentPlace.country].filter(Boolean).join(', ');
  $('current-temp').innerHTML = `${Math.round(current.temperature_2m)}<small>°</small>`;
  $('current-icon').textContent = icon;
  $('current-label').textContent = label;
  $('current-range').textContent = `Today ${Math.round(data.daily.temperature_2m_min[0])}° / ${Math.round(data.daily.temperature_2m_max[0])}°`;
  $('feels').textContent = `${Math.round(current.apparent_temperature)}°`;
  $('wind').textContent = `${Math.round(current.wind_speed_10m)} km/h`;
  $('humidity').textContent = `${current.relative_humidity_2m}%`;
  $('timezone').textContent = data.timezone_abbreviation || data.timezone;
  $('sunrise').textContent = formatTime(data.daily.sunrise[0]);
  $('sunset').textContent = formatTime(data.daily.sunset[0]);
  renderSunProgress(data.daily.sunrise[0], data.daily.sunset[0], current.time);
  renderHourly(data.hourly, current.time);
  renderDaily(data.daily);
}

function renderHourly(hourly, now) {
  const start = Math.max(0, hourly.time.findIndex((time) => time >= now));
  const fragment = document.createDocumentFragment();
  for (let i = start; i < Math.min(start + 12, hourly.time.length); i += 1) {
    const [label, icon] = describe(hourly.weather_code[i]);
    const item = document.createElement('div');
    item.className = 'hour';
    item.innerHTML = `<time>${i === start ? 'Now' : formatTime(hourly.time[i])}</time><div class="weather-icon" title="${label}">${icon}</div><strong>${Math.round(hourly.temperature_2m[i])}°</strong><small>${hourly.precipitation_probability[i]}% rain</small>`;
    fragment.append(item);
  }
  $('hours').replaceChildren(fragment);
}

function renderDaily(daily) {
  const fragment = document.createDocumentFragment();
  daily.time.forEach((date, i) => {
    const [label, icon] = describe(daily.weather_code[i]);
    const item = document.createElement('div');
    item.className = 'day';
    item.innerHTML = `<time>${i === 0 ? 'Today' : new Intl.DateTimeFormat(undefined, { weekday: 'short' }).format(new Date(`${date}T12:00:00`))}</time><div class="weather-icon" title="${label}">${icon}</div><div class="temps"><b>${Math.round(daily.temperature_2m_max[i])}°</b><span>${Math.round(daily.temperature_2m_min[i])}°</span></div><small>${daily.precipitation_probability_max[i] || 0}% rain</small>`;
    fragment.append(item);
  });
  $('days').replaceChildren(fragment);
}

async function renderMap(place) {
  const mapEl = $('map');
  map.radarPath = null;
  map.radarTime = null;
  const center = project(place.latitude, place.longitude, map.zoom);
  map.centerX = mapEl.clientWidth / 2 - center.x;
  map.centerY = mapEl.clientHeight / 2 - center.y;
  drawMap();
  $('map-status').textContent = 'Map · drag to explore';
  $('map-caption').textContent = 'Live precipitation radar when available';
  try {
    const response = await fetch('https://api.rainviewer.com/public/weather-maps.json');
    if (!response.ok) throw new Error();
    const radar = await response.json();
    const latest = radar.radar?.past?.at(-1);
    if (!latest) throw new Error();
    map.radarPath = latest.path;
    map.radarTime = latest.time;
    drawRadar();
    $('map-caption').textContent = `Radar updated ${new Date(latest.time * 1000).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}`;
  } catch {
    $('radar-layer').replaceChildren();
    $('map-caption').textContent = 'Live radar unavailable · map still interactive';
  }
}

function drawMap() {
  const layer = $('tile-layer');
  const size = 256;
  const scale = 2 ** map.zoom;
  const tileX = Math.floor((map.centerX + $('map').clientWidth / 2) / size);
  const tileY = Math.floor((map.centerY + $('map').clientHeight / 2) / size);
  const cols = Math.ceil($('map').clientWidth / size) + 2;
  const rows = Math.ceil($('map').clientHeight / size) + 2;
  const fragment = document.createDocumentFragment();
  for (let x = tileX - 1; x < tileX + cols; x += 1) {
    for (let y = tileY - 1; y < tileY + rows; y += 1) {
      if (y < 0 || y >= scale) continue;
      const tile = document.createElement('img');
      tile.className = 'tile';
      tile.alt = '';
      tile.src = `https://tile.openstreetmap.org/${map.zoom}/${((x % scale) + scale) % scale}/${y}.png`;
      tile.style.left = `${x * size + map.centerX}px`;
      tile.style.top = `${y * size + map.centerY}px`;
      fragment.append(tile);
    }
  }
  layer.replaceChildren(fragment);
  if (map.radarPath) drawRadar();
  const point = project(currentPlace.latitude, currentPlace.longitude, map.zoom);
  $('map-marker').style.left = `${point.x + map.centerX}px`;
  $('map-marker').style.top = `${point.y + map.centerY}px`;
}

function drawRadar() {
  const layer = $('radar-layer');
  const size = 256;
  const scale = 2 ** map.zoom;
  const tileX = Math.floor((map.centerX + $('map').clientWidth / 2) / size);
  const tileY = Math.floor((map.centerY + $('map').clientHeight / 2) / size);
  const fragment = document.createDocumentFragment();
  for (let x = tileX - 1; x < tileX + 5; x += 1) {
    for (let y = tileY - 1; y < tileY + 4; y += 1) {
      if (y < 0 || y >= scale) continue;
      const tile = document.createElement('img');
      tile.className = 'radar-tile';
      tile.alt = '';
      tile.src = `https://tilecache.rainviewer.com${map.radarPath}/256/${map.zoom}/${((x % scale) + scale) % scale}/${y}/2/1_1.png`;
      tile.style.left = `${x * size + map.centerX}px`;
      tile.style.top = `${y * size + map.centerY}px`;
      tile.dataset.timestamp = map.radarTime;
      fragment.append(tile);
    }
  }
  layer.replaceChildren(fragment);
}

function project(latitude, longitude, zoom) {
  const scale = 256 * 2 ** zoom;
  const x = (longitude + 180) / 360 * scale;
  const sin = Math.sin(latitude * Math.PI / 180);
  const y = (0.5 - Math.log((1 + sin) / (1 - sin)) / (4 * Math.PI)) * scale;
  return { x, y };
}

function changeZoom(delta) {
  map.zoom = Math.max(3, Math.min(11, map.zoom + delta));
  renderMap(currentPlace);
}

function startPan(event) {
  map.dragging = true; map.startX = event.clientX; map.startY = event.clientY;
  map.baseX = map.centerX; map.baseY = map.centerY; $('map').classList.add('dragging');
}
function pan(event) {
  if (!map.dragging) return;
  map.centerX = map.baseX + event.clientX - map.startX;
  map.centerY = map.baseY + event.clientY - map.startY;
  drawMap();
}
function endPan() { map.dragging = false; $('map').classList.remove('dragging'); }

function renderSunProgress(sunrise, sunset, now) {
  const start = new Date(sunrise).getTime(), end = new Date(sunset).getTime(), current = new Date(now).getTime();
  const progress = Math.max(0, Math.min(1, (current - start) / (end - start)));
  $('sun-progress').style.strokeDasharray = `${progress * 100} 100`;
  const angle = Math.PI - progress * Math.PI;
  $('sun-dot').setAttribute('cx', `${200 + Math.cos(angle) * 180}`);
  $('sun-dot').setAttribute('cy', `${103 - Math.sin(angle) * 90}`);
}

function describe(code) { return weatherCodes[code] || ['Variable conditions', '🌤️']; }
function formatTime(value) { return new Date(value).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }); }
function setLoading() { $('notice').classList.remove('visible'); $('dashboard').hidden = true; $('weather').hidden = false; }
function showError(message) { $('weather').hidden = true; $('dashboard').hidden = true; $('notice').textContent = message; $('notice').classList.add('visible'); }
