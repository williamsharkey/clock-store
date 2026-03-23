import { randomUUID } from 'node:crypto'
import { createServer } from 'node:http'
import { URL } from 'node:url'
import { WebSocketServer } from 'ws'
import { propagate, twoline2satrec } from 'satellite.js'

const host = process.env.HOST || '0.0.0.0'
const port = Number(process.env.PORT || 8788)
const players = new Map()

const TLE_SOURCES = [
  {
    key: 'stations',
    label: 'Stations',
    url: 'https://celestrak.org/NORAD/elements/gp.php?GROUP=STATIONS',
    trailSegments: 24,
    trailSpanMinutes: 360,
    keepNames: true,
  },
  {
    key: 'starlink',
    label: 'Starlink',
    url: 'https://celestrak.org/NORAD/elements/gp.php?GROUP=STARLINK',
    trailSegments: 5,
    trailSpanMinutes: 45,
    keepNames: false,
  },
  {
    key: 'oneweb',
    label: 'OneWeb',
    url: 'https://celestrak.org/NORAD/elements/gp.php?GROUP=ONEWEB',
    trailSegments: 6,
    trailSpanMinutes: 75,
    keepNames: false,
  },
  {
    key: 'cosmos2251',
    label: 'Cosmos 2251 Debris',
    url: 'https://celestrak.org/NORAD/elements/gp.php?NAME=COSMOS%202251%20DEB',
    trailSegments: 4,
    trailSpanMinutes: 45,
    keepNames: false,
  },
  {
    key: 'iridium33',
    label: 'Iridium 33 Debris',
    url: 'https://celestrak.org/NORAD/elements/gp.php?NAME=IRIDIUM%2033%20DEB',
    trailSegments: 4,
    trailSpanMinutes: 45,
    keepNames: false,
  },
  {
    key: 'fengyun1c',
    label: 'Fengyun 1C Debris',
    url: 'https://celestrak.org/NORAD/elements/gp.php?NAME=FENGYUN%201C%20DEB',
    trailSegments: 4,
    trailSpanMinutes: 45,
    keepNames: false,
  },
]

const BODY_SPECS = [
  { key: 'moon', label: 'Moon', command: '301', spanDays: 8, step: '4 h' },
  { key: 'mercury', label: 'Mercury', command: '199', spanDays: 180, step: '5 d' },
  { key: 'venus', label: 'Venus', command: '299', spanDays: 180, step: '5 d' },
  { key: 'mars', label: 'Mars', command: '499', spanDays: 220, step: '7 d' },
  { key: 'phobos', label: 'Phobos', command: '401', spanDays: 40, step: '1 d' },
  { key: 'deimos', label: 'Deimos', command: '402', spanDays: 60, step: '2 d' },
  { key: 'jupiter', label: 'Jupiter', command: '599', spanDays: 260, step: '10 d' },
  { key: 'io', label: 'Io', command: '501', spanDays: 30, step: '12 h' },
  { key: 'europa', label: 'Europa', command: '502', spanDays: 40, step: '12 h' },
  { key: 'ganymede', label: 'Ganymede', command: '503', spanDays: 50, step: '12 h' },
  { key: 'callisto', label: 'Callisto', command: '504', spanDays: 70, step: '18 h' },
  { key: 'saturn', label: 'Saturn', command: '699', spanDays: 320, step: '12 d' },
  { key: 'titan', label: 'Titan', command: '606', spanDays: 80, step: '1 d' },
  { key: 'uranus', label: 'Uranus', command: '799', spanDays: 420, step: '16 d' },
  { key: 'titania', label: 'Titania', command: '703', spanDays: 120, step: '2 d' },
  { key: 'oberon', label: 'Oberon', command: '704', spanDays: 140, step: '2 d' },
  { key: 'neptune', label: 'Neptune', command: '899', spanDays: 520, step: '20 d' },
  { key: 'triton', label: 'Triton', command: '801', spanDays: 180, step: '3 d' },
]

let apiCache = {
  expiresAt: 0,
  payload: null,
}

async function fetchTextWithRetry(url, attempts = 3) {
  let lastError = null
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    const response = await fetch(url)
    if (response.ok) {
      return response.text()
    }
    lastError = new Error(`Request failed: ${response.status}`)
    await new Promise((resolve) => setTimeout(resolve, 350 * (attempt + 1)))
  }
  throw lastError ?? new Error('Request failed')
}

async function fetchJsonWithRetry(url, attempts = 3) {
  let lastError = null
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    const response = await fetch(url)
    if (response.ok) {
      return response.json()
    }
    lastError = new Error(`Request failed: ${response.status}`)
    await new Promise((resolve) => setTimeout(resolve, 350 * (attempt + 1)))
  }
  throw lastError ?? new Error('Request failed')
}

function roundKm(value) {
  return Math.round(value)
}

function parseTleText(text) {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trimEnd())
    .filter(Boolean)
  const objects = []

  for (let index = 0; index < lines.length - 2; index += 3) {
    const name = lines[index]
    const line1 = lines[index + 1]
    const line2 = lines[index + 2]
    if (!line1?.startsWith('1 ') || !line2?.startsWith('2 ')) continue
    objects.push({ name: name.trim(), line1: line1.trim(), line2: line2.trim() })
  }

  return objects
}

function toVector(point) {
  if (!point) return null
  return [roundKm(point.x), roundKm(point.y), roundKm(point.z)]
}

function buildTrail(satrec, now, config) {
  const trail = []
  const steps = Math.max(2, config.trailSegments)
  const spanMinutes = config.trailSpanMinutes

  for (let step = 0; step < steps; step += 1) {
    const offsetMinutes = -spanMinutes / 2 + (spanMinutes * step) / (steps - 1)
    const date = new Date(now.getTime() + offsetMinutes * 60_000)
    const state = propagate(satrec, date)
    if (!state.position) return null
    trail.push(toVector(state.position))
  }

  return trail
}

async function fetchSatelliteCategory(config, now) {
  const text = await fetchTextWithRetry(config.url)
  const records = parseTleText(text)
  const objects = []

  for (const record of records) {
    const satrec = twoline2satrec(record.line1, record.line2)
    const current = propagate(satrec, now)
    if (!current.position) continue
    const trail = buildTrail(satrec, now, config)
    if (!trail) continue
    objects.push({
      ...(config.keepNames ? { name: record.name } : {}),
      position: toVector(current.position),
      trail,
    })
  }

  return {
    key: config.key,
    label: config.label,
    total: records.length,
    rendered: objects.length,
    objects,
  }
}

function formatHorizonsDate(date) {
  const pad = (value) => String(value).padStart(2, '0')
  return `${date.getUTCFullYear()}-${pad(date.getUTCMonth() + 1)}-${pad(date.getUTCDate())} ${pad(date.getUTCHours())}:${pad(date.getUTCMinutes())}`
}

function parseHorizonsVectors(result) {
  const start = result.indexOf('$$SOE')
  const end = result.indexOf('$$EOE')
  if (start === -1 || end === -1 || end <= start) return []

  const lines = result
    .slice(start + 5, end)
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)

  const vectors = []
  for (let index = 0; index < lines.length - 1; index += 1) {
    if (!lines[index].includes('= A.D.')) continue
    const coords = lines[index + 1]
    const match = coords.match(/X =([^\s]+)\s+Y =([^\s]+)\s+Z =([^\s]+)/)
    if (!match) continue
    vectors.push([
      roundKm(Number(match[1])),
      roundKm(Number(match[2])),
      roundKm(Number(match[3])),
    ])
  }

  return vectors
}

async function fetchBodyVectors(spec, now) {
  const start = new Date(now.getTime() - spec.spanDays * 12 * 60 * 60 * 1000)
  const stop = new Date(now.getTime() + spec.spanDays * 12 * 60 * 60 * 1000)
  const params = new URLSearchParams({
    format: 'json',
    COMMAND: `'${spec.command}'`,
    MAKE_EPHEM: "'YES'",
    EPHEM_TYPE: "'VECTORS'",
    CENTER: "'500@399'",
    START_TIME: `'${formatHorizonsDate(start)}'`,
    STOP_TIME: `'${formatHorizonsDate(stop)}'`,
    STEP_SIZE: `'${spec.step}'`,
  })

  const payload = await fetchJsonWithRetry(`https://ssd.jpl.nasa.gov/api/horizons.api?${params.toString()}`)
  const trail = parseHorizonsVectors(payload.result || '')
  const midpoint = trail[Math.floor(trail.length / 2)] || null
  return {
    key: spec.key,
    label: spec.label,
    position: midpoint,
    trail,
  }
}

async function buildSpacePayload() {
  const now = new Date()
  const satellites = await Promise.all(TLE_SOURCES.map((config) => fetchSatelliteCategory(config, now)))
  const bodies = []
  for (const spec of BODY_SPECS) {
    bodies.push(await fetchBodyVectors(spec, now))
  }

  return {
    generatedAt: now.toISOString(),
    satellites,
    bodies,
  }
}

async function getSpacePayload() {
  if (apiCache.payload && Date.now() < apiCache.expiresAt) {
    return apiCache.payload
  }

  const payload = await buildSpacePayload()
  apiCache = {
    payload,
    expiresAt: Date.now() + 15 * 60 * 1000,
  }
  return payload
}

function serializePlayers() {
  return [...players.values()].map((player) => ({
    id: player.id,
    name: player.name,
    faceSeed: player.faceSeed,
    position: player.position,
    yaw: player.yaw,
    boarded: player.boarded,
    stage: player.stage,
    updatedAt: player.updatedAt,
  }))
}

function broadcast(payload, exceptId = null) {
  const message = JSON.stringify(payload)
  for (const player of players.values()) {
    if (exceptId && player.id === exceptId) continue
    if (player.socket.readyState === 1) {
      player.socket.send(message)
    }
  }
}

function pruneStalePlayers() {
  const cutoff = Date.now() - 15_000
  let changed = false
  for (const [id, player] of players.entries()) {
    if (player.updatedAt < cutoff || player.socket.readyState >= 2) {
      players.delete(id)
      changed = true
    }
  }
  if (changed) {
    broadcast({ type: 'snapshot', players: serializePlayers() })
  }
}

function sendJson(response, statusCode, payload) {
  response.writeHead(statusCode, {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Content-Type': 'application/json; charset=utf-8',
  })
  response.end(JSON.stringify(payload))
}

const server = createServer(async (request, response) => {
  if (!request.url) {
    sendJson(response, 400, { error: 'Missing URL' })
    return
  }

  if (request.method === 'OPTIONS') {
    response.writeHead(204, {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    })
    response.end()
    return
  }

  const url = new URL(request.url, `http://${request.headers.host || 'localhost'}`)

  if (url.pathname === '/health') {
    sendJson(response, 200, { ok: true })
    return
  }

  if (url.pathname === '/api/space') {
    try {
      const payload = await getSpacePayload()
      sendJson(response, 200, payload)
    } catch (error) {
      sendJson(response, 500, {
        error: 'Failed to build live space payload',
        detail: error instanceof Error ? error.message : String(error),
      })
    }
    return
  }

  sendJson(response, 404, { error: 'Not found' })
})

const wss = new WebSocketServer({ noServer: true })

server.on('upgrade', (request, socket, head) => {
  const url = new URL(request.url || '/', `http://${request.headers.host || 'localhost'}`)
  if (url.pathname !== '/ws') {
    socket.destroy()
    return
  }

  wss.handleUpgrade(request, socket, head, (upgradedSocket) => {
    wss.emit('connection', upgradedSocket, request)
  })
})

wss.on('connection', (socket) => {
  const id = randomUUID()
  const player = {
    id,
    socket,
    name: `Hurlonaut-${id.slice(0, 4)}`,
    faceSeed: Math.floor(Math.random() * 1_000_000_000),
    position: [0, 2, 0],
    yaw: 0,
    boarded: false,
    stage: 'arrival',
    updatedAt: Date.now(),
  }

  players.set(id, player)
  socket.send(JSON.stringify({ type: 'welcome', id, players: serializePlayers() }))
  broadcast({ type: 'snapshot', players: serializePlayers() }, id)

  socket.on('message', (raw) => {
    let payload
    try {
      payload = JSON.parse(String(raw))
    } catch {
      return
    }

    player.updatedAt = Date.now()

    if (payload.type === 'hello') {
      if (typeof payload.name === 'string' && payload.name.trim()) {
        player.name = payload.name.trim().slice(0, 32)
      }
      if (typeof payload.faceSeed === 'number') {
        player.faceSeed = payload.faceSeed
      }
      if (typeof payload.stage === 'string') {
        player.stage = payload.stage.slice(0, 32)
      }
      broadcast({ type: 'snapshot', players: serializePlayers() })
      return
    }

    if (payload.type === 'state') {
      if (Array.isArray(payload.position) && payload.position.length === 3) {
        player.position = payload.position.map((value) => Number(value) || 0)
      }
      if (typeof payload.yaw === 'number') {
        player.yaw = payload.yaw
      }
      player.boarded = Boolean(payload.boarded)
      if (typeof payload.stage === 'string') {
        player.stage = payload.stage.slice(0, 32)
      }
      broadcast({
        type: 'state',
        player: {
          id: player.id,
          name: player.name,
          faceSeed: player.faceSeed,
          position: player.position,
          yaw: player.yaw,
          boarded: player.boarded,
          stage: player.stage,
          updatedAt: player.updatedAt,
        },
      }, id)
      return
    }

    if (payload.type === 'barf') {
      broadcast({
        type: 'barf',
        id,
        position: player.position,
        yaw: player.yaw,
        boarded: player.boarded,
        stage: player.stage,
        at: Date.now(),
      }, id)
    }
  })

  socket.on('close', () => {
    players.delete(id)
    broadcast({ type: 'leave', id })
  })
})

setInterval(pruneStalePlayers, 5000)

server.listen(port, host, () => {
  console.log(`Clock Store gateway listening on http://${host}:${port} and ws://${host}:${port}/ws`)
})
