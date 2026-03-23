import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { BARF_FACTS } from './barfFacts.ts'
import { CONTINENT_LOOPS } from './earthContours.ts'

const LINE_COLOR = 0x42ff6b
const EARTH_RADIUS = 8.5
const EARTH_KM = 6371
const STORE_POS = new THREE.Vector3(-18, 0, 0)
const CLERK_POS = new THREE.Vector3(-12, 0, 5)
const EARTH_POS = new THREE.Vector3(10, 12, 0)
const CAMERA_LERP = 0.08

type StageId =
  | 'arrival'
  | 'clerk'
  | 'satellites'
  | 'planets'
  | 'ride'
  | 'riders'
  | 'heresy'

type SatelliteFocus = 'fleets' | 'junk' | 'all'
type PlanetFocus = 'moon' | 'outer' | 'all'

interface StoryState {
  stage: StageId
  hasBag: boolean
  satelliteFocus: SatelliteFocus
  planetFocus: PlanetFocus
}

interface StageCard {
  copy: string
  choices: [string, string, string]
}

type Vec3Tuple = [number, number, number]

interface SpaceObject {
  name?: string
  position: Vec3Tuple
  trail: Vec3Tuple[]
}

interface SatelliteCategory {
  key: string
  label: string
  total: number
  rendered: number
  objects: SpaceObject[]
}

interface SpaceBody {
  key: string
  label: string
  position: Vec3Tuple | null
  trail: Vec3Tuple[]
}

interface SpacePayload {
  generatedAt: string
  satellites: SatelliteCategory[]
  bodies: SpaceBody[]
}

interface RemotePlayerState {
  id: string
  faceSeed: number
  position: Vec3Tuple
  yaw: number
  boarded: boolean
  stage?: string
}

interface RemoteAvatar {
  group: THREE.Group
  targetPosition: THREE.Vector3
  yaw: number
  boarded: boolean
}

interface BarfParticle {
  position: THREE.Vector3
  velocity: THREE.Vector3
  life: number
  size: number
}

const BARF_SEGMENTS_PER_PARTICLE = 4

const INITIAL_STORY: StoryState = {
  stage: 'arrival',
  hasBag: false,
  satelliteFocus: 'fleets',
  planetFocus: 'moon',
}

function mulberry32(seed: number) {
  let value = seed >>> 0
  return () => {
    value += 0x6d2b79f5
    let t = value
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function latLonToVector3(longitude: number, latitude: number, radius: number) {
  const lat = THREE.MathUtils.degToRad(latitude)
  const lon = THREE.MathUtils.degToRad(longitude)
  const x = radius * Math.cos(lat) * Math.cos(lon)
  const y = radius * Math.sin(lat)
  const z = -radius * Math.cos(lat) * Math.sin(lon)
  return new THREE.Vector3(x, y, z)
}

function buildLineFromPoints(points: THREE.Vector3[], closed = false, opacity = 1) {
  const geometry = new THREE.BufferGeometry().setFromPoints(points)
  const material = new THREE.LineBasicMaterial({
    color: LINE_COLOR,
    transparent: opacity < 1,
    opacity,
  })
  return closed ? new THREE.LineLoop(geometry, material) : new THREE.Line(geometry, material)
}

function buildWireframe(geometry: THREE.BufferGeometry, opacity = 1) {
  const wire = new THREE.WireframeGeometry(geometry)
  return new THREE.LineSegments(
    wire,
    new THREE.LineBasicMaterial({
      color: LINE_COLOR,
      transparent: opacity < 1,
      opacity,
    }),
  )
}

function buildCross(position: THREE.Vector3, size: number) {
  const group = new THREE.Group()
  const segments = [
    [new THREE.Vector3(-size, 0, 0), new THREE.Vector3(size, 0, 0)],
    [new THREE.Vector3(0, -size, 0), new THREE.Vector3(0, size, 0)],
    [new THREE.Vector3(0, 0, -size), new THREE.Vector3(0, 0, size)],
  ]

  for (const [a, b] of segments) {
    const line = buildLineFromPoints([a, b])
    group.add(line)
  }

  group.position.copy(position)
  return group
}

function buildOrbitRing(radius: number, tilt = 0, segments = 80) {
  const points: THREE.Vector3[] = []
  for (let index = 0; index < segments; index += 1) {
    const angle = (index / segments) * Math.PI * 2
    points.push(new THREE.Vector3(Math.cos(angle) * radius, 0, Math.sin(angle) * radius))
  }
  const ring = buildLineFromPoints(points, true, 0.35)
  ring.rotation.x = tilt
  return ring
}

function createWireFace(seed: number) {
  const rng = mulberry32(seed)
  const group = new THREE.Group()
  const eyeY = 0.1 + rng() * 0.08
  const eyeSpread = 0.18 + rng() * 0.06
  const mouthWidth = 0.16 + rng() * 0.18
  const mouthDrop = -0.18 - rng() * 0.08

  const leftEye = buildLineFromPoints([
    new THREE.Vector3(-eyeSpread - 0.07, eyeY, 0.39),
    new THREE.Vector3(-eyeSpread + 0.07, eyeY + (rng() - 0.5) * 0.05, 0.39),
  ])
  const rightEye = buildLineFromPoints([
    new THREE.Vector3(eyeSpread - 0.07, eyeY + (rng() - 0.5) * 0.05, 0.39),
    new THREE.Vector3(eyeSpread + 0.07, eyeY, 0.39),
  ])
  const nose = buildLineFromPoints([
    new THREE.Vector3(0, 0.02, 0.39),
    new THREE.Vector3(0.04 - rng() * 0.08, -0.08, 0.39),
  ])
  const mouth = buildLineFromPoints([
    new THREE.Vector3(-mouthWidth, mouthDrop, 0.39),
    new THREE.Vector3(0, mouthDrop - (rng() - 0.5) * 0.08, 0.39),
    new THREE.Vector3(mouthWidth, mouthDrop, 0.39),
  ])

  group.add(leftEye, rightEye, nose, mouth)
  return group
}

function createWireCharacter(seed: number, scale = 1) {
  const group = new THREE.Group()
  const torso = buildWireframe(new THREE.BoxGeometry(0.95, 1.35, 0.55))
  torso.position.y = 1.7
  const head = buildWireframe(new THREE.BoxGeometry(0.78, 0.78, 0.78))
  head.position.y = 2.72
  head.add(createWireFace(seed))

  const hips = buildWireframe(new THREE.BoxGeometry(0.7, 0.28, 0.4), 0.8)
  hips.position.y = 0.95

  const leftLeg = buildWireframe(new THREE.BoxGeometry(0.26, 1, 0.26))
  leftLeg.position.set(-0.2, 0.47, 0)
  const rightLeg = buildWireframe(new THREE.BoxGeometry(0.26, 1, 0.26))
  rightLeg.position.set(0.2, 0.47, 0)

  const leftArm = buildWireframe(new THREE.BoxGeometry(0.26, 1.05, 0.26))
  leftArm.position.set(-0.73, 1.65, 0)
  leftArm.rotation.z = 0.28

  const rightArm = buildWireframe(new THREE.BoxGeometry(0.26, 1.05, 0.26))
  rightArm.position.set(0.73, 1.65, 0)
  rightArm.rotation.z = -0.42

  const shoulderBar = buildWireframe(new THREE.BoxGeometry(1.35, 0.18, 0.2), 0.8)
  shoulderBar.position.y = 2.15

  const rightHandAnchor = new THREE.Object3D()
  rightHandAnchor.position.set(1.05, 1.35, 0.1)
  group.add(rightHandAnchor)

  const mouthAnchor = new THREE.Object3D()
  mouthAnchor.position.set(0.12, 2.65, 0.55)
  group.add(mouthAnchor)

  group.add(torso, head, hips, leftLeg, rightLeg, leftArm, rightArm, shoulderBar)
  group.scale.setScalar(scale)

  return { group, rightHandAnchor, mouthAnchor }
}

function createStore() {
  const group = new THREE.Group()

  const shell = buildWireframe(new THREE.BoxGeometry(12, 7, 10))
  shell.position.set(STORE_POS.x, 3.5, STORE_POS.z)

  const awning = buildWireframe(new THREE.BoxGeometry(8, 1.2, 2.5))
  awning.position.set(STORE_POS.x + 1, 4.6, STORE_POS.z + 6.2)

  const door = buildWireframe(new THREE.BoxGeometry(2.2, 4.1, 0.15))
  door.position.set(STORE_POS.x + 2.5, 2.1, STORE_POS.z + 5.08)

  const windowFrame = buildWireframe(new THREE.BoxGeometry(3.8, 2.4, 0.15))
  windowFrame.position.set(STORE_POS.x - 2.7, 3.1, STORE_POS.z + 5.08)

  const roof = buildWireframe(new THREE.ConeGeometry(7.6, 2.4, 4))
  roof.position.set(STORE_POS.x, 8, STORE_POS.z)
  roof.rotation.y = Math.PI / 4

  const signFrame = buildWireframe(new THREE.BoxGeometry(7.5, 1.4, 0.12), 0.75)
  signFrame.position.set(STORE_POS.x, 6.1, STORE_POS.z + 5.18)

  group.add(shell, awning, door, windowFrame, roof, signFrame)
  return group
}

function createClerk() {
  const { group, rightHandAnchor, mouthAnchor } = createWireCharacter(40404, 1.18)
  group.position.copy(CLERK_POS)
  group.rotation.y = -0.32

  const cigarette = buildWireframe(new THREE.BoxGeometry(0.42, 0.05, 0.05))
  cigarette.position.set(0.22, 0, 0)
  cigarette.rotation.z = 0.15
  rightHandAnchor.add(cigarette)

  const ember = buildCross(new THREE.Vector3(0.46, 0, 0), 0.03)
  cigarette.add(ember)

  return { group, mouthAnchor }
}

function createSmokeLines(count: number) {
  return Array.from({ length: count }, () =>
    buildLineFromPoints([
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(0, 0.2, 0),
      new THREE.Vector3(0, 0.4, 0),
      new THREE.Vector3(0, 0.6, 0),
    ], false, 0.4),
  )
}

function createEarthRig() {
  const earthGroup = new THREE.Group()
  earthGroup.position.copy(EARTH_POS)

  const pedestal = buildWireframe(new THREE.CylinderGeometry(3.4, 4.1, 7.5, 10))
  pedestal.position.y = -EARTH_RADIUS + 1.2
  earthGroup.add(pedestal)

  const spinGroup = new THREE.Group()
  spinGroup.rotation.z = THREE.MathUtils.degToRad(23.4)
  earthGroup.add(spinGroup)

  const globe = buildWireframe(new THREE.IcosahedronGeometry(EARTH_RADIUS, 2))
  spinGroup.add(globe)

  for (const latitude of [-60, -30, 0, 30, 60]) {
    const points: THREE.Vector3[] = []
    for (let segment = 0; segment < 80; segment += 1) {
      const longitude = (segment / 80) * 360
      points.push(latLonToVector3(longitude, latitude, EARTH_RADIUS * 1.001))
    }
    spinGroup.add(buildLineFromPoints(points, true, 0.28))
  }

  for (const longitude of [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]) {
    const points: THREE.Vector3[] = []
    for (let segment = 0; segment <= 40; segment += 1) {
      const latitude = -78 + (segment / 40) * 156
      points.push(latLonToVector3(longitude, latitude, EARTH_RADIUS * 1.001))
    }
    spinGroup.add(buildLineFromPoints(points, false, 0.2))
  }

  for (const loop of CONTINENT_LOOPS) {
    const points = loop.map(([longitude, latitude]) => latLonToVector3(longitude, latitude, EARTH_RADIUS * 1.02))
    spinGroup.add(buildLineFromPoints(points, true))
  }

  const mount = new THREE.Group()
  mount.position.y = EARTH_RADIUS + 0.45
  spinGroup.add(mount)

  const mountStem = buildWireframe(new THREE.CylinderGeometry(0.18, 0.18, 1.1, 6))
  mountStem.position.y = -0.55
  mount.add(mountStem)

  const mountCap = buildWireframe(new THREE.CylinderGeometry(0.5, 0.5, 0.24, 8))
  mount.add(mountCap)

  const hourPivot = new THREE.Group()
  const minutePivot = new THREE.Group()
  const secondPivot = new THREE.Group()
  mount.add(hourPivot)
  hourPivot.add(minutePivot)
  minutePivot.add(secondPivot)

  const hourLength = 4.6
  const minuteLength = 3.2
  const secondLength = 2.4

  const hourArm = buildWireframe(new THREE.BoxGeometry(hourLength, 0.2, 0.2))
  hourArm.position.x = hourLength / 2
  hourPivot.add(hourArm)

  const minuteArm = buildWireframe(new THREE.BoxGeometry(minuteLength, 0.16, 0.16))
  minuteArm.position.x = minuteLength / 2
  minutePivot.position.x = hourLength
  minutePivot.add(minuteArm)

  const secondArm = buildWireframe(new THREE.BoxGeometry(secondLength, 0.12, 0.12))
  secondArm.position.x = secondLength / 2
  secondPivot.position.x = minuteLength
  secondPivot.add(secondArm)

  const seatAnchor = new THREE.Object3D()
  seatAnchor.position.x = secondLength
  secondPivot.add(seatAnchor)

  const seat = new THREE.Group()
  const seatBase = buildWireframe(new THREE.BoxGeometry(0.68, 0.18, 0.68))
  const seatBack = buildWireframe(new THREE.BoxGeometry(0.68, 0.85, 0.12))
  seatBack.position.set(-0.12, 0.48, -0.28)
  const seatStrap = buildWireframe(new THREE.TorusGeometry(0.33, 0.04, 5, 14), 0.8)
  seatStrap.rotation.x = Math.PI / 2
  seatStrap.position.y = 0.38
  seat.add(seatBase, seatBack, seatStrap)
  seatAnchor.add(seat)

  const bag = buildWireframe(new THREE.CylinderGeometry(0.22, 0.16, 0.5, 5))
  bag.position.set(-0.28, -0.18, 0.28)
  seat.add(bag)

  const moonPivot = new THREE.Group()
  spinGroup.add(moonPivot)
  moonPivot.add(buildOrbitRing(EARTH_RADIUS + 5.2, 0.22, 60))
  const moon = buildWireframe(new THREE.IcosahedronGeometry(1.15, 0))
  moon.position.set(EARTH_RADIUS + 5.2, 0.8, 0)
  moonPivot.add(moon)

  return {
    earthGroup,
    spinGroup,
    hourPivot,
    minutePivot,
    secondPivot,
    seatAnchor,
    moonPivot,
    moon,
    bag,
  }
}

function createStars(count: number) {
  const vertices: number[] = []
  const rng = mulberry32(918273)
  for (let index = 0; index < count; index += 1) {
    const radius = 120 + rng() * 150
    const theta = rng() * Math.PI * 2
    const phi = Math.acos(1 - rng() * 2)
    const position = new THREE.Vector3(
      radius * Math.sin(phi) * Math.cos(theta),
      radius * Math.cos(phi),
      radius * Math.sin(phi) * Math.sin(theta),
    )
    vertices.push(
      position.x - 0.2, position.y, position.z, position.x + 0.2, position.y, position.z,
      position.x, position.y - 0.2, position.z, position.x, position.y + 0.2, position.z,
    )
  }
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3))
  return new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color: LINE_COLOR, transparent: true, opacity: 0.3 }))
}

function buildBarfSystem(maxParticles: number) {
  const geometry = new THREE.BufferGeometry()
  const positions = new Float32Array(maxParticles * BARF_SEGMENTS_PER_PARTICLE * 2 * 3)
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  geometry.setDrawRange(0, 0)

  const material = new THREE.LineBasicMaterial({ color: LINE_COLOR, transparent: true, opacity: 0.95 })
  const lines = new THREE.LineSegments(geometry, material)
  lines.frustumCulled = false

  return { lines, positions }
}

function createSpaceCloudGroup(payload: SpacePayload) {
  const root = new THREE.Group()
  const stations = new THREE.Group()
  const fleets = new THREE.Group()
  const junk = new THREE.Group()
  root.add(stations, fleets, junk)

  const addTrailSet = (parent: THREE.Group, objects: SpaceObject[], scale: (value: Vec3Tuple) => THREE.Vector3, pointSize: number) => {
    const trailVertices: number[] = []
    const crossVertices: number[] = []

    for (const object of objects) {
      let previous: THREE.Vector3 | null = null
      for (const point of object.trail) {
        const next = scale(point)
        if (previous) {
          trailVertices.push(previous.x, previous.y, previous.z, next.x, next.y, next.z)
        }
        previous = next
      }

      const center = scale(object.position)
      crossVertices.push(
        center.x - pointSize, center.y, center.z, center.x + pointSize, center.y, center.z,
        center.x, center.y - pointSize, center.z, center.x, center.y + pointSize, center.z,
        center.x, center.y, center.z - pointSize, center.x, center.y, center.z + pointSize,
      )
    }

    const trailGeometry = new THREE.BufferGeometry()
    trailGeometry.setAttribute('position', new THREE.Float32BufferAttribute(trailVertices, 3))
    const crossesGeometry = new THREE.BufferGeometry()
    crossesGeometry.setAttribute('position', new THREE.Float32BufferAttribute(crossVertices, 3))

    parent.add(
      new THREE.LineSegments(trailGeometry, new THREE.LineBasicMaterial({ color: LINE_COLOR, transparent: true, opacity: 0.3 })),
      new THREE.LineSegments(crossesGeometry, new THREE.LineBasicMaterial({ color: LINE_COLOR })),
    )
  }

  const scaleSatellite = (point: Vec3Tuple) => new THREE.Vector3(
    (point[0] / EARTH_KM) * EARTH_RADIUS,
    (point[1] / EARTH_KM) * EARTH_RADIUS,
    (point[2] / EARTH_KM) * EARTH_RADIUS,
  )

  const scaleDeepSpace = (point: Vec3Tuple) => {
    const vector = new THREE.Vector3(point[0], point[1], point[2])
    const distance = vector.length()
    if (distance < 1) return new THREE.Vector3()
    const radius = EARTH_RADIUS * 15 + Math.log10(distance / 1000 + 1) * 10
    return vector.normalize().multiplyScalar(radius)
  }

  const stationObjects: SpaceObject[] = []
  const fleetObjects: SpaceObject[] = []
  const junkObjects: SpaceObject[] = []

  for (const category of payload.satellites) {
    if (category.key === 'stations') {
      stationObjects.push(...category.objects)
    } else if (category.key === 'starlink' || category.key === 'oneweb') {
      fleetObjects.push(...category.objects)
    } else {
      junkObjects.push(...category.objects)
    }
  }

  addTrailSet(stations, stationObjects, scaleSatellite, 0.18)
  addTrailSet(fleets, fleetObjects, scaleSatellite, 0.08)
  addTrailSet(junk, junkObjects, scaleSatellite, 0.06)

  const moonGroup = new THREE.Group()
  const outerGroup = new THREE.Group()
  root.add(moonGroup, outerGroup)

  addTrailSet(
    moonGroup,
    payload.bodies.filter((body) => body.key === 'moon' && body.position).map((body) => ({
      position: body.position as Vec3Tuple,
      trail: body.trail,
    })),
    scaleDeepSpace,
    0.8,
  )

  addTrailSet(
    outerGroup,
    payload.bodies.filter((body) => body.key !== 'moon' && body.position).map((body) => ({
      position: body.position as Vec3Tuple,
      trail: body.trail,
    })),
    scaleDeepSpace,
    0.7,
  )

  stations.name = 'stations'
  fleets.name = 'fleets'
  junk.name = 'junk'
  moonGroup.name = 'moon'
  outerGroup.name = 'outer'

  return { root, stations, fleets, junk, moonGroup, outerGroup }
}

function averageVectors(vectors: THREE.Vector3[]) {
  if (!vectors.length) return new THREE.Vector3()
  const result = new THREE.Vector3()
  for (const vector of vectors) result.add(vector)
  return result.multiplyScalar(1 / vectors.length)
}

function getStageCard(story: StoryState): StageCard {
  switch (story.stage) {
    case 'clerk':
      return {
        copy: '"hey buddy wanna buy a clock," says the smoking wireframe clerk, as if the pitch explains the planet-sized machine.',
        choices: ['Buy the clock', 'Ask about sidereal time', 'Back to the curb'],
      }
    case 'satellites':
      return {
        copy: story.satelliteFocus === 'junk'
          ? 'Real debris clouds carve green scars around Earth: Cosmos 2251, Iridium 33, Fengyun 1C, all still drawing consequences.'
          : story.satelliteFocus === 'all'
            ? 'Live fleets, stations, and junk all stack together into one humming orbital bruise.'
            : 'Real stations, Starlink, and OneWeb tracks wrap the Earth in live traffic.',
        choices: ['Focus fleets and stations', 'Focus junk clouds', story.hasBag ? 'Strap into the clock' : 'Return to store'],
      }
    case 'planets':
      return {
        copy: story.planetFocus === 'moon'
          ? 'The Moon gets the close-up, but the rest of the solar crowd is still there in compressed, real JPL vectors.'
          : story.planetFocus === 'outer'
            ? 'Planets and their moons arc across the sky in live trajectories, compressed so one scene can hold the mess.'
            : 'Moon, planets, and major moons all draw their current paths around your ride.',
        choices: ['Track the Moon', 'Track planets and moons', story.hasBag ? 'Back to the clock' : 'Back to store'],
      }
    case 'ride':
      return {
        copy: 'You are strapped to the end of the second hand. The hour arm turns once an hour, the minute arm once a minute, the second arm once a second. The bag is in your lap.',
        choices: ['Hold the ride line', 'Watch planets and moons', 'Look for other riders'],
      }
    case 'riders':
      return {
        copy: 'Other players blink in over the gateway with random faces, their own doomed trajectories, and no visible intention to keep lunch down.',
        choices: ['Stay with the riders', 'Watch live space traffic', 'Unstrap to store'],
      }
    case 'heresy':
      return {
        copy: 'He says a sidereal day is 23:56:04, brings up Bruno and Hypatia, and acts offended that philosophy once got people killed over celestial positions.',
        choices: ['Stay in the heresy pitch', 'Buy the clock anyway', 'Back to store'],
      }
    case 'arrival':
    default:
      return {
        copy: 'A low-poly smoker guards the Clock Store. Beside him, a wireframe Earth carries a clock on its tilted crown while live orbital traffic crawls around it.',
        choices: ['Talk to the smoker', 'See live satellite traffic', 'See planets and moons'],
      }
  }
}

function getGatewayUrls() {
  const raw = (import.meta.env.VITE_CLOCKSTORE_WS_URL || '').trim()
  const fallbackProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const fallbackHost = window.location.hostname || 'localhost'
  const base = raw || `${fallbackProtocol}//${fallbackHost}:8788/ws`
  const wsUrl = new URL(base, window.location.href)
  if (wsUrl.pathname === '/' || wsUrl.pathname === '') {
    wsUrl.pathname = '/ws'
  }
  const apiUrl = new URL(wsUrl.toString().replace(/^ws/, 'http'))
  const wsPath = wsUrl.pathname.endsWith('/ws')
    ? wsUrl.pathname.slice(0, -3)
    : wsUrl.pathname
  apiUrl.pathname = `${wsPath.replace(/\/$/, '') || ''}/api/space`
  return { wsUrl: wsUrl.toString(), apiUrl: apiUrl.toString() }
}

function disposeScene(root: THREE.Object3D) {
  root.traverse((object) => {
    const meshLike = object as THREE.Mesh
    if (meshLike.geometry) meshLike.geometry.dispose()
    const material = (meshLike as THREE.Line | THREE.Mesh).material
    if (Array.isArray(material)) {
      material.forEach((entry) => entry.dispose())
    } else if (material) {
      material.dispose()
    }
  })
}

export default function Scene() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const storyRef = useRef(INITIAL_STORY)
  const [story, setStory] = useState(INITIAL_STORY)
  const [barfLine, setBarfLine] = useState('Press B once the bag is yours.')
  const [aside, setAside] = useState('Three buttons. One bag. No settings.')
  const [peers, setPeers] = useState(1)
  const [spaceReady, setSpaceReady] = useState(false)
  const [spaceNote, setSpaceNote] = useState('Drawing the sky…')

  const stageCard = useMemo(() => getStageCard(story), [story])

  useEffect(() => {
    storyRef.current = story
  }, [story])

  const choose = (index: 1 | 2 | 3) => {
    setStory((current) => {
      const next = { ...current }

      switch (current.stage) {
        case 'arrival':
          if (index === 1) next.stage = 'clerk'
          if (index === 2) {
            next.stage = 'satellites'
            next.satelliteFocus = 'fleets'
          }
          if (index === 3) {
            next.stage = 'planets'
            next.planetFocus = 'moon'
          }
          break
        case 'clerk':
          if (index === 1) {
            next.stage = 'ride'
            next.hasBag = true
            setAside('He hands you a wireframe barf bag and buckles the second-hand seat.')
          }
          if (index === 2) next.stage = 'heresy'
          if (index === 3) next.stage = 'arrival'
          break
        case 'satellites':
          if (index === 1) {
            next.satelliteFocus = 'fleets'
            setAside('Stations, Starlink, and OneWeb stay in view.')
          }
          if (index === 2) {
            next.satelliteFocus = 'junk'
            setAside('Debris clouds take the whole stage.')
          }
          if (index === 3) {
            next.stage = next.hasBag ? 'ride' : 'arrival'
          }
          break
        case 'planets':
          if (index === 1) {
            next.planetFocus = 'moon'
            setAside('The Moon gets the closest camera.')
          }
          if (index === 2) {
            next.planetFocus = 'outer'
            setAside('Outer planets and major moons take over the sky.')
          }
          if (index === 3) {
            next.stage = next.hasBag ? 'ride' : 'arrival'
          }
          break
        case 'ride':
          if (index === 1) {
            next.stage = 'ride'
            setAside('You stay locked into the second-hand seat.')
          }
          if (index === 2) {
            next.stage = 'planets'
            next.planetFocus = 'all'
          }
          if (index === 3) next.stage = 'riders'
          break
        case 'riders':
          if (index === 1) {
            next.stage = 'riders'
            setAside('Everybody looks like a bad idea rendered perfectly.')
          }
          if (index === 2) {
            next.stage = 'satellites'
            next.satelliteFocus = 'all'
          }
          if (index === 3) next.stage = 'arrival'
          break
        case 'heresy':
          if (index === 1) {
            next.stage = 'heresy'
            setAside('He keeps talking about Bruno, Hypatia, and how clocks are political.')
          }
          if (index === 2) {
            next.stage = 'ride'
            next.hasBag = true
          }
          if (index === 3) next.stage = 'arrival'
          break
      }

      return next
    })
  }

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const { wsUrl, apiUrl } = getGatewayUrls()
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: false })
    renderer.setPixelRatio(1)
    renderer.setSize(window.innerWidth, window.innerHeight, false)
    renderer.setClearColor(0x000000, 1)

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1200)
    scene.add(camera)

    const root = new THREE.Group()
    scene.add(root)
    root.add(createStars(160))

    const ground = buildWireframe(new THREE.PlaneGeometry(160, 160, 24, 24), 0.24)
    ground.rotation.x = -Math.PI / 2
    root.add(ground)

    const store = createStore()
    root.add(store)

    const clerk = createClerk()
    root.add(clerk.group)
    const smokeLines = createSmokeLines(4)
    smokeLines.forEach((line) => root.add(line))

    const earthRig = createEarthRig()
    root.add(earthRig.earthGroup)

    const sun = buildWireframe(new THREE.IcosahedronGeometry(2, 1), 0.8)
    sun.position.set(-20, 40, -75)
    root.add(sun)

    const planetOrbits = new THREE.Group()
    planetOrbits.add(buildOrbitRing(18, 0.2), buildOrbitRing(26, 0.35), buildOrbitRing(34, -0.15))
    planetOrbits.position.copy(EARTH_POS)
    root.add(planetOrbits)

    const orbitLayers = new THREE.Group()
    orbitLayers.position.copy(EARTH_POS)
    root.add(orbitLayers)

    let spaceClouds: ReturnType<typeof createSpaceCloudGroup> | null = null
    let spacePayload: SpacePayload | null = null

    const remotePlayers = new Map<string, RemoteAvatar>()
    const particles: BarfParticle[] = []
    const barfSystem = buildBarfSystem(900)
    root.add(barfSystem.lines)

    const currentCameraPos = new THREE.Vector3(-6, 16, 34)
    const currentTarget = new THREE.Vector3(-4, 9, 4)
    const desiredCameraPos = new THREE.Vector3()
    const desiredTarget = new THREE.Vector3()
    const seatWorld = new THREE.Vector3()
    const seatPrev = new THREE.Vector3()
    const seatVelocity = new THREE.Vector3(1, 0, 0)
    const seatDrift = new THREE.Vector3()
    const earthWorld = new THREE.Vector3()
    const moonWorld = new THREE.Vector3()
    const stageAnchor = new THREE.Vector3()
    const tmp = new THREE.Vector3()
    const tmp2 = new THREE.Vector3()
    const forward = new THREE.Vector3()
    const coneDir = new THREE.Vector3()
    const emitterOrigin = new THREE.Vector3()
    const emitterVelocity = new THREE.Vector3()
    const mouthOrigin = new THREE.Vector3()
    const cameraBarfOrigin = new THREE.Vector3()

    let animationId = 0
    let socket: WebSocket | null = null
    let localId: string | null = null
    let disposed = false
    let lastStateSend = 0
    let lastPeerCount = 1

    const localSeedKey = 'clock-store-face-seed'
    const storedSeed = Number(window.localStorage.getItem(localSeedKey) || 0)
    const localFaceSeed = Number.isFinite(storedSeed) && storedSeed > 0 ? storedSeed : Math.floor(Math.random() * 1_000_000_000)
    if (!storedSeed) {
      window.localStorage.setItem(localSeedKey, String(localFaceSeed))
    }

    function playerOffset(seed: number) {
      const angle = ((seed % 360) * Math.PI) / 180
      return new THREE.Vector3(Math.cos(angle) * 1.2, 0, Math.sin(angle) * 1.2)
    }

    function createRemoteAvatar(faceSeed: number) {
      const avatar = createWireCharacter(faceSeed, 0.82)
      const bag = buildWireframe(new THREE.CylinderGeometry(0.12, 0.09, 0.28, 5), 0.75)
      bag.position.set(-0.22, 0.95, 0.24)
      avatar.group.add(bag)
      return avatar.group
    }

    function upsertRemotePlayer(player: RemotePlayerState) {
      if (localId && player.id === localId) return
      const target = new THREE.Vector3(player.position[0], player.position[1], player.position[2])
      const existing = remotePlayers.get(player.id)
      if (existing) {
        existing.targetPosition.copy(target)
        existing.yaw = player.yaw
        existing.boarded = player.boarded
        return
      }

      const group = createRemoteAvatar(player.faceSeed)
      group.position.copy(target)
      root.add(group)
      remotePlayers.set(player.id, {
        group,
        targetPosition: target,
        yaw: player.yaw,
        boarded: player.boarded,
      })
    }

    function syncRemotePlayers(playersState: RemotePlayerState[]) {
      const incoming = new Set(playersState.map((player) => player.id))
      for (const player of playersState) upsertRemotePlayer(player)
      for (const [id, avatar] of remotePlayers.entries()) {
        if (incoming.has(id)) continue
        root.remove(avatar.group)
        remotePlayers.delete(id)
      }
      const nextPeers = remotePlayers.size + 1
      if (nextPeers !== lastPeerCount) {
        lastPeerCount = nextPeers
        setPeers(nextPeers)
      }
    }

    function emitBarf(
      origin: THREE.Vector3,
      direction: THREE.Vector3,
      carrierVelocity: THREE.Vector3,
      strength = 1,
      nearStart = 0.6,
    ) {
      const rng = mulberry32(Math.floor(origin.x * 1000 + origin.y * 400 + origin.z * 200))
      const normalized = direction.clone().normalize()
      const carry = carrierVelocity.clone()
      for (let index = 0; index < 72; index += 1) {
        const coneAngle = 0.12 + rng() * 0.38
        const coneSpin = rng() * Math.PI * 2
        const sideways = new THREE.Vector3(
          Math.cos(coneSpin) * Math.sin(coneAngle),
          Math.sin(coneSpin) * Math.sin(coneAngle) * 0.75,
          Math.cos(coneAngle),
        )
        coneDir.copy(sideways).applyQuaternion(
          new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), normalized),
        )
        const velocity = coneDir
          .clone()
          .multiplyScalar(8 + rng() * 10 * strength)
          .add(carry.clone().multiplyScalar(0.55))
          .add(new THREE.Vector3((rng() - 0.5) * 0.8, (rng() - 0.5) * 0.6, (rng() - 0.5) * 0.8))
        particles.push({
          position: origin.clone().addScaledVector(normalized, nearStart + rng() * 0.45),
          velocity,
          life: 2.8 + rng() * 2.6,
          size: 0.12 + rng() * 0.18,
        })
      }
    }

    function triggerBarf(fromRemote = false, origin?: THREE.Vector3, yaw?: number) {
      if (!fromRemote && !storyRef.current.hasBag) {
        setAside('You do not own the bag yet.')
        return
      }
      const fact = BARF_FACTS[Math.floor(Math.random() * BARF_FACTS.length)]
      if (!fromRemote) {
        setBarfLine(fact)
      }

      camera.getWorldDirection(forward)
      const boarded = storyRef.current.stage === 'ride' || storyRef.current.stage === 'riders' || (storyRef.current.stage === 'planets' && storyRef.current.hasBag)
      if (fromRemote) {
        emitterOrigin.copy(origin ?? seatWorld).add(boarded ? new THREE.Vector3(0, 0.9, 0) : new THREE.Vector3(0, 1.2, 0))
        forward.set(Math.sin(yaw ?? 0), 0.18, Math.cos(yaw ?? 0)).normalize()
        emitterVelocity.set(0, 0, 0)
      } else {
        mouthOrigin.copy(origin ?? seatWorld).add(boarded ? new THREE.Vector3(0, 0.9, 0) : new THREE.Vector3(0, 1.2, 0))
        cameraBarfOrigin.copy(camera.position).addScaledVector(forward, 0.78).add(new THREE.Vector3(0, -0.08, 0))
        emitterOrigin.copy(mouthOrigin).lerp(cameraBarfOrigin, boarded ? 0.84 : 0.5)
        emitterVelocity.copy(seatDrift)
      }
      if (!storyRef.current.hasBag && !fromRemote) return
      emitBarf(emitterOrigin, forward, emitterVelocity, fromRemote ? 0.8 : 1.1, fromRemote ? 0.6 : 0.16)

      if (!fromRemote && socket?.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: 'barf' }))
      }
    }

    function handleKey(event: KeyboardEvent) {
      if (event.repeat) return
      if (event.key === '1') choose(1)
      if (event.key === '2') choose(2)
      if (event.key === '3') choose(3)
      if (event.key.toLowerCase() === 'b') {
        triggerBarf()
      }
    }

    function getLocalStagePosition(state: StoryState) {
      const offset = playerOffset(localFaceSeed)
      if (state.stage === 'clerk' || state.stage === 'heresy') {
        return CLERK_POS.clone().add(new THREE.Vector3(-1.4, 0, 2.6)).add(offset)
      }
      if (state.stage === 'satellites' || state.stage === 'planets') {
        return EARTH_POS.clone().add(new THREE.Vector3(-8, -EARTH_RADIUS, 10)).add(offset)
      }
      return STORE_POS.clone().add(new THREE.Vector3(10, 0, 18)).add(offset)
    }

    function updateOrbitVisibility(state: StoryState) {
      if (!spaceClouds) return
      spaceClouds.stations.visible = state.stage === 'arrival' || state.stage === 'ride' || state.stage === 'satellites'
      spaceClouds.fleets.visible = state.stage === 'satellites' && (state.satelliteFocus === 'fleets' || state.satelliteFocus === 'all')
      spaceClouds.junk.visible = state.stage === 'satellites' && (state.satelliteFocus === 'junk' || state.satelliteFocus === 'all')
      if (state.stage === 'arrival' || state.stage === 'ride') {
        spaceClouds.stations.visible = true
        spaceClouds.fleets.visible = state.stage === 'ride'
        spaceClouds.junk.visible = false
      }
      spaceClouds.moonGroup.visible = state.stage === 'planets' || state.stage === 'ride' || state.stage === 'arrival'
      spaceClouds.outerGroup.visible = state.stage === 'planets' && state.planetFocus !== 'moon'
      if (state.stage === 'planets' && state.planetFocus === 'all') {
        spaceClouds.moonGroup.visible = true
        spaceClouds.outerGroup.visible = true
      }
      if (state.stage === 'ride') {
        spaceClouds.moonGroup.visible = true
        spaceClouds.outerGroup.visible = false
      }
      if (state.stage === 'arrival') {
        spaceClouds.moonGroup.visible = false
        spaceClouds.outerGroup.visible = false
      }
    }

    function updateCamera(nowSeconds: number) {
      const state = storyRef.current
      earthRig.seatAnchor.getWorldPosition(seatWorld)
      earthRig.moon.getWorldPosition(moonWorld)
      earthRig.earthGroup.getWorldPosition(earthWorld)

      tmp.copy(seatWorld).sub(seatPrev)
      seatDrift.copy(tmp).multiplyScalar(60)
      if (tmp.lengthSq() > 0.000001) {
        seatVelocity.copy(tmp).normalize()
      }
      seatPrev.copy(seatWorld)

      const boarded = state.stage === 'ride' || state.stage === 'riders' || (state.stage === 'planets' && state.hasBag)
      earthRig.bag.visible = state.hasBag

      if (boarded) {
        const outward = tmp2.copy(seatWorld).sub(earthWorld).normalize()
        desiredCameraPos.copy(seatWorld)
        desiredTarget.copy(seatWorld)
        if (state.stage === 'ride') {
          desiredCameraPos.addScaledVector(outward, 2.4).add(new THREE.Vector3(0, 1.2, 0)).addScaledVector(seatVelocity, -1.5)
          desiredTarget.addScaledVector(seatVelocity, 8).addScaledVector(outward, 1.4)
        } else if (state.stage === 'riders') {
          const remotePositions = [...remotePlayers.values()].map((avatar) => avatar.group.position)
          const riderFocus = remotePositions.length ? averageVectors(remotePositions) : moonWorld
          desiredCameraPos.addScaledVector(outward, 6).add(new THREE.Vector3(0, 3.5, 0))
          desiredTarget.copy(riderFocus)
        } else {
          desiredCameraPos.addScaledVector(outward, 6).add(new THREE.Vector3(0, 3.5, 0))
          desiredTarget.copy(state.planetFocus === 'moon' ? moonWorld : earthWorld.clone().add(new THREE.Vector3(-20, 24, -40)))
        }
        stageAnchor.copy(seatWorld)
      } else {
        stageAnchor.copy(getLocalStagePosition(state))
        if (state.stage === 'clerk') {
          desiredCameraPos.set(-18, 6.2, 14)
          desiredTarget.copy(CLERK_POS).add(new THREE.Vector3(0, 2.3, 0))
        } else if (state.stage === 'satellites') {
          desiredCameraPos.copy(EARTH_POS).add(new THREE.Vector3(-24, 14, 24))
          desiredTarget.copy(EARTH_POS).add(new THREE.Vector3(0, 5, 0))
        } else if (state.stage === 'planets') {
          desiredCameraPos.copy(EARTH_POS).add(new THREE.Vector3(22, 19, 26))
          desiredTarget.copy(state.planetFocus === 'moon' ? moonWorld : EARTH_POS.clone().add(new THREE.Vector3(0, 20, -60)))
        } else if (state.stage === 'heresy') {
          desiredCameraPos.set(-8, 11, 24)
          desiredTarget.copy(EARTH_POS).multiplyScalar(0.55).add(CLERK_POS.clone().multiplyScalar(0.45))
        } else {
          desiredCameraPos.set(-6, 16 + Math.sin(nowSeconds * 0.15) * 0.8, 34)
          desiredTarget.set(-4, 9, 4)
        }
      }

      currentCameraPos.lerp(desiredCameraPos, CAMERA_LERP)
      currentTarget.lerp(desiredTarget, CAMERA_LERP)
      camera.position.copy(currentCameraPos)
      camera.lookAt(currentTarget)
    }

    function updateSmoke(nowSeconds: number) {
      clerk.mouthAnchor.getWorldPosition(tmp)
      for (const [index, line] of smokeLines.entries()) {
        const points: THREE.Vector3[] = []
        for (let step = 0; step < 7; step += 1) {
          const t = step / 6
          const wobble = nowSeconds * 1.3 + index * 1.1 + t * 4
          points.push(new THREE.Vector3(
            tmp.x + Math.sin(wobble) * 0.16 * t,
            tmp.y + 0.3 + t * 1.9,
            tmp.z + Math.cos(wobble * 1.1) * 0.16 * t,
          ))
        }
        ;(line.geometry as THREE.BufferGeometry).setFromPoints(points)
      }
    }

    function updateClock(nowSeconds: number) {
      earthRig.spinGroup.rotation.y = nowSeconds / 40
      earthRig.hourPivot.rotation.y = nowSeconds * ((Math.PI * 2) / 3600)
      earthRig.minutePivot.rotation.y = nowSeconds * ((Math.PI * 2) / 60)
      earthRig.secondPivot.rotation.y = nowSeconds * Math.PI * 2
      earthRig.moonPivot.rotation.y = nowSeconds * 0.22
      sun.rotation.y = nowSeconds * 0.06
    }

    function updateRemoteAvatars() {
      for (const avatar of remotePlayers.values()) {
        avatar.group.position.lerp(avatar.targetPosition, 0.18)
        avatar.group.rotation.y += (avatar.yaw - avatar.group.rotation.y) * 0.16
      }
    }

    function updateBarf(dt: number) {
      const positions = barfSystem.positions
      let drawCount = 0
      for (let index = particles.length - 1; index >= 0; index -= 1) {
        const particle = particles[index]
        particle.life -= dt
        if (particle.life <= 0) {
          particles.splice(index, 1)
          continue
        }
        particle.velocity.y -= 3.5 * dt
        particle.velocity.multiplyScalar(0.992)
        particle.position.addScaledVector(particle.velocity, dt)
        const tip = particle.position
        const tail = particle.position.clone().addScaledVector(particle.velocity, -0.045)
        const size = particle.size
        const base = drawCount * BARF_SEGMENTS_PER_PARTICLE * 6
        positions[base] = tail.x
        positions[base + 1] = tail.y
        positions[base + 2] = tail.z
        positions[base + 3] = tip.x
        positions[base + 4] = tip.y
        positions[base + 5] = tip.z

        positions[base + 6] = tip.x - size
        positions[base + 7] = tip.y
        positions[base + 8] = tip.z
        positions[base + 9] = tip.x + size
        positions[base + 10] = tip.y
        positions[base + 11] = tip.z

        positions[base + 12] = tip.x
        positions[base + 13] = tip.y - size
        positions[base + 14] = tip.z
        positions[base + 15] = tip.x
        positions[base + 16] = tip.y + size
        positions[base + 17] = tip.z

        positions[base + 18] = tip.x
        positions[base + 19] = tip.y
        positions[base + 20] = tip.z - size
        positions[base + 21] = tip.x
        positions[base + 22] = tip.y
        positions[base + 23] = tip.z + size
        drawCount += 1
      }
      const attribute = barfSystem.lines.geometry.getAttribute('position') as THREE.BufferAttribute
      attribute.needsUpdate = true
      barfSystem.lines.geometry.setDrawRange(0, drawCount * BARF_SEGMENTS_PER_PARTICLE * 2)
    }

    async function loadSpaceData() {
      try {
        const response = await fetch(apiUrl)
        if (!response.ok) throw new Error(`space payload failed: ${response.status}`)
        const payload = (await response.json()) as SpacePayload
        if (spaceClouds) {
          orbitLayers.remove(spaceClouds.root)
          disposeScene(spaceClouds.root)
        }
        spacePayload = payload
        spaceClouds = createSpaceCloudGroup(payload)
        orbitLayers.add(spaceClouds.root)
        updateOrbitVisibility(storyRef.current)
        setSpaceReady(true)
        setSpaceNote('Live trajectories drawn from the gateway.')
      } catch (error) {
        setSpaceReady(false)
        setSpaceNote(error instanceof Error ? error.message : String(error))
      }
    }

    function connectSocket() {
      socket = new WebSocket(wsUrl)

      socket.addEventListener('open', () => {
        socket?.send(JSON.stringify({
          type: 'hello',
          faceSeed: localFaceSeed,
          name: `Rider-${String(localFaceSeed).slice(-4)}`,
          stage: storyRef.current.stage,
        }))
      })

      socket.addEventListener('message', (event) => {
        let payload: unknown
        try {
          payload = JSON.parse(event.data)
        } catch {
          return
        }

        if (!payload || typeof payload !== 'object') return
        const message = payload as Record<string, unknown>
        if (message.type === 'welcome') {
          localId = typeof message.id === 'string' ? message.id : null
          const playersState = Array.isArray(message.players) ? (message.players as RemotePlayerState[]) : []
          syncRemotePlayers(playersState)
          return
        }
        if (message.type === 'snapshot') {
          const playersState = Array.isArray(message.players) ? (message.players as RemotePlayerState[]) : []
          syncRemotePlayers(playersState)
          return
        }
        if (message.type === 'state' && message.player && typeof message.player === 'object') {
          upsertRemotePlayer(message.player as RemotePlayerState)
          const nextPeers = remotePlayers.size + 1
          if (nextPeers !== lastPeerCount) {
            lastPeerCount = nextPeers
            setPeers(nextPeers)
          }
          return
        }
        if (message.type === 'leave' && typeof message.id === 'string') {
          const existing = remotePlayers.get(message.id)
          if (existing) {
            root.remove(existing.group)
            remotePlayers.delete(message.id)
            lastPeerCount = remotePlayers.size + 1
            setPeers(lastPeerCount)
          }
          return
        }
        if (message.type === 'barf' && Array.isArray(message.position)) {
          const position = message.position as Vec3Tuple
          triggerBarf(
            true,
            new THREE.Vector3(position[0], position[1] + 1, position[2]),
            typeof message.yaw === 'number' ? message.yaw : 0,
          )
        }
      })

      socket.addEventListener('close', () => {
        if (!disposed) {
          window.setTimeout(connectSocket, 1200)
        }
      })
    }

    function resize() {
      const width = window.innerWidth
      const height = window.innerHeight
      camera.aspect = width / height
      camera.updateProjectionMatrix()
      renderer.setSize(width, height, false)
    }

    function animate() {
      if (disposed) return
      animationId = window.requestAnimationFrame(animate)
      const now = performance.now() / 1000
      const dt = 1 / 60

      updateClock(now)
      updateSmoke(now)
      updateOrbitVisibility(storyRef.current)
      updateCamera(now)
      updateRemoteAvatars()
      updateBarf(dt)

      if (socket?.readyState === WebSocket.OPEN && now - lastStateSend > 0.18) {
        const state = storyRef.current
        const boarded = state.stage === 'ride' || state.stage === 'riders' || (state.stage === 'planets' && state.hasBag)
        const position = boarded ? seatWorld : getLocalStagePosition(state)
        socket.send(JSON.stringify({
          type: 'state',
          position: [Number(position.x.toFixed(2)), Number(position.y.toFixed(2)), Number(position.z.toFixed(2))],
          yaw: Number(camera.rotation.y.toFixed(3)),
          boarded,
          stage: state.stage,
        }))
        lastStateSend = now
      }

      renderer.render(scene, camera)
    }

    window.addEventListener('resize', resize)
    window.addEventListener('keydown', handleKey)
    connectSocket()
    void loadSpaceData()
    const refreshTimer = window.setInterval(() => {
      void loadSpaceData()
    }, 15 * 60 * 1000)
    animate()

    return () => {
      disposed = true
      window.cancelAnimationFrame(animationId)
      window.removeEventListener('resize', resize)
      window.removeEventListener('keydown', handleKey)
      window.clearInterval(refreshTimer)
      socket?.close()
      disposeScene(scene)
      renderer.dispose()
    }
  }, [])

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative', background: '#000' }}>
      <canvas
        ref={canvasRef}
        style={{
          width: '100%',
          height: '100%',
          display: 'block',
          imageRendering: 'pixelated',
          cursor: 'crosshair',
        }}
      />
      <div
        style={{
          position: 'absolute',
          inset: 0,
          pointerEvents: 'none',
          color: '#42ff6b',
          fontFamily: '"Courier New", monospace',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
          padding: '18px',
          textTransform: 'uppercase',
          letterSpacing: '0.08em',
        }}
      >
        <div style={{ maxWidth: 760 }}>
          <div style={{ fontSize: 16, opacity: 0.9 }}>Clock Store</div>
          <div style={{ fontSize: 28, lineHeight: 1.2, marginTop: 10 }}>{stageCard.copy}</div>
          <div style={{ fontSize: 13, opacity: 0.82, marginTop: 10 }}>{aside}</div>
          <div style={{ fontSize: 12, opacity: 0.65, marginTop: 8 }}>
            {spaceReady ? spaceNote : `The live sky is still drawing itself in. ${spaceNote}`}
          </div>
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', gap: 16, flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', pointerEvents: 'auto' }}>
            {[1, 2, 3].map((index) => (
              <button
                key={index}
                onClick={() => choose(index as 1 | 2 | 3)}
                style={{
                  minWidth: 220,
                  border: '1px solid #42ff6b',
                  background: 'rgba(0,0,0,0.9)',
                  color: '#42ff6b',
                  padding: '14px 16px',
                  textAlign: 'left',
                  fontFamily: '"Courier New", monospace',
                  fontSize: 15,
                  textTransform: 'uppercase',
                  letterSpacing: '0.08em',
                  cursor: 'pointer',
                }}
              >
                <div style={{ fontSize: 12, opacity: 0.75 }}>{index}</div>
                <div style={{ marginTop: 6 }}>{stageCard.choices[index - 1]}</div>
              </button>
            ))}
          </div>

          <div style={{ maxWidth: 420, textAlign: 'right' }}>
            <div style={{ fontSize: 14 }}>B: Barf</div>
            <div style={{ fontSize: 12, opacity: 0.78, marginTop: 8 }}>{barfLine}</div>
            <div style={{ fontSize: 12, opacity: 0.55, marginTop: 8 }}>{peers} rider{peers === 1 ? '' : 's'} in the wireframe mess</div>
          </div>
        </div>
      </div>
    </div>
  )
}
