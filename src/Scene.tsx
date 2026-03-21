import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { WebGPURenderer, MeshStandardNodeMaterial } from 'three/webgpu'
import { uv, uniform, uniformArray, float, floor, fract, uint, Fn } from 'three/tsl'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { Pane } from 'tweakpane'

const SEGMENTS_PER_TURN = 12
const DAYS_PER_YEAR = 365
const DAY_SEGS_PER_TURN = 12
const HOURS_PER_DAY = 24
const HOUR_SEGS_PER_TURN = 8

const paletteColors = [
  new THREE.Color('#e63946'),
  new THREE.Color('#f4a261'),
  new THREE.Color('#e9c46a'),
  new THREE.Color('#2a9d8f'),
  new THREE.Color('#457b9d'),
  new THREE.Color('#1d3557'),
  new THREE.Color('#a8dadc'),
  new THREE.Color('#6a4c93'),
  new THREE.Color('#f72585'),
  new THREE.Color('#4cc9f0'),
  new THREE.Color('#80b918'),
  new THREE.Color('#ffffff'),
]

const colorPalette = uniformArray(paletteColors)
const numTurnsUniform = uniform(5)

const coilColorNode = Fn(() => {
  const turnFraction = fract(uv().x.mul(numTurnsUniform))
  const segmentIndex = uint(floor(turnFraction.mul(float(SEGMENTS_PER_TURN))))
  return colorPalette.element(segmentIndex)
})()

class HelixCurve extends THREE.Curve<THREE.Vector3> {
  turns: number
  radius: number
  length: number

  constructor(turns: number, radius: number, length: number) {
    super()
    this.turns = turns
    this.radius = radius
    this.length = length
  }

  getPoint(t: number, target = new THREE.Vector3()) {
    const angle = t * this.turns * Math.PI * 2
    return target.set(
      Math.cos(angle) * this.radius,
      (t - 0.5) * this.length,
      Math.sin(angle) * this.radius,
    )
  }
}

// Writes a day-coil position into arr[j..j+2] using the analytical helix Frenet frame.
function writeDayPos(
  t: number, omega: number, R: number, L: number, tMag: number,
  totalDayTurns: number, dayOff: number,
  arr: Float32Array, j: number,
) {
  const theta = t * omega
  const cT = Math.cos(theta), sT = Math.sin(theta)
  const px = cT * R, py = (t - 0.5) * L, pz = sT * R
  // Frenet normal (toward axis), ny=0
  const nx = -cT, nz = -sT
  // Normalized tangent
  const tx = -sT * R * omega / tMag, ty = L / tMag, tz = cT * R * omega / tMag
  // Binormal = T x N  (ny=0 simplifies)
  const bx = ty * nz, by = tz * nx - tx * nz, bz = -ty * nx
  // Day-coil offset
  const dA = t * totalDayTurns * Math.PI * 2
  const cD = Math.cos(dA), sD = Math.sin(dA)
  arr[j]     = px + dayOff * (cD * nx + sD * bx)
  arr[j + 1] = py + dayOff * sD * by
  arr[j + 2] = pz + dayOff * (cD * nz + sD * bz)
}

interface Runtime {
  renderer: WebGPURenderer
  scene: THREE.Scene
  camera: THREE.PerspectiveCamera
  controls: OrbitControls
  coilMesh: THREE.Mesh | null
  dayCoilLine: THREE.Line | null
  hourCoilLine: THREE.Line | null
  material: MeshStandardNodeMaterial
  lineMaterial: THREE.LineBasicMaterial
  pane: Pane
}

export default function Scene() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const runtimeRef = useRef<Runtime | null>(null)
  const rafRef = useRef<number>(0)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    let disposed = false

    const params = {
      turns: 2,
      tubeRadius: 0.08,
      coilRadius: 1,
      turnSpacing: 5,
      outerOffset: 0.05,
      hourOffset: 0.01,
    }

    function rebuild(rt: Runtime) {
      if (rt.coilMesh) { rt.scene.remove(rt.coilMesh); rt.coilMesh.geometry.dispose() }
      if (rt.dayCoilLine) { rt.scene.remove(rt.dayCoilLine); rt.dayCoilLine.geometry.dispose() }
      if (rt.hourCoilLine) { rt.scene.remove(rt.hourCoilLine); rt.hourCoilLine.geometry.dispose() }

      const numTurns = Math.max(1, params.turns)
      const length = numTurns * params.turnSpacing
      numTurnsUniform.value = numTurns
      const R = params.coilRadius
      const L = length
      const omega = numTurns * Math.PI * 2
      const tMag = Math.sqrt(R * R * omega * omega + L * L)
      const totalDayTurns = numTurns * DAYS_PER_YEAR
      const dayOff = params.tubeRadius + params.outerOffset
      const tmpColor = new THREE.Color()

      // ====== BIG COIL (tube) ======
      const curve = new HelixCurve(numTurns, R, L)
      const tubSegs = Math.ceil(numTurns * SEGMENTS_PER_TURN) * 8
      rt.coilMesh = new THREE.Mesh(
        new THREE.TubeGeometry(curve, tubSegs, params.tubeRadius, 12, false),
        rt.material,
      )
      rt.scene.add(rt.coilMesh)

      // ====== DAY COIL (line) ======
      const totalDaySegs = totalDayTurns * DAY_SEGS_PER_TURN
      const totalDayPts = totalDaySegs + 1
      const dayPos = new Float32Array(totalDayPts * 3)
      const dayCol = new Float32Array(totalDayPts * 3)

      for (let i = 0; i < totalDayPts; i++) {
        const t = i / totalDaySegs
        const j = i * 3
        writeDayPos(t, omega, R, L, tMag, totalDayTurns, dayOff, dayPos, j)
        const hue = (t * numTurns) % 1
        tmpColor.setHSL(hue, 1, 0.5)
        dayCol[j] = tmpColor.r; dayCol[j + 1] = tmpColor.g; dayCol[j + 2] = tmpColor.b
      }

      const dayGeom = new THREE.BufferGeometry()
      dayGeom.setAttribute('position', new THREE.BufferAttribute(dayPos, 3))
      dayGeom.setAttribute('color', new THREE.BufferAttribute(dayCol, 3))
      rt.dayCoilLine = new THREE.Line(dayGeom, rt.lineMaterial)
      rt.scene.add(rt.dayCoilLine)

      // ====== HOUR COIL (line) ======
      const totalHourTurns = totalDayTurns * HOURS_PER_DAY
      const totalHourSegs = totalHourTurns * HOUR_SEGS_PER_TURN
      const totalHourPts = totalHourSegs + 1

      // Pass 1: day-coil positions at hour resolution
      const dayHR = new Float32Array(totalHourPts * 3)
      for (let i = 0; i < totalHourPts; i++) {
        writeDayPos(i / totalHourSegs, omega, R, L, tMag, totalDayTurns, dayOff, dayHR, i * 3)
      }

      // Pass 2: parallel-transport frame along day-coil path + hour offsets
      const hourPos = new Float32Array(totalHourPts * 3)
      const hourCol = new Float32Array(totalHourPts * 3)

      const prevTan = new THREE.Vector3()
      const curTan = new THREE.Vector3()
      const dN = new THREE.Vector3()
      const dB = new THREE.Vector3()
      const crossV = new THREE.Vector3()
      const rotMat = new THREE.Matrix4()
      const hOff = params.hourOffset

      // Seed tangent from first two day-coil points
      curTan.set(dayHR[3] - dayHR[0], dayHR[4] - dayHR[1], dayHR[5] - dayHR[2]).normalize()

      // Seed normal perpendicular to tangent
      const ref = Math.abs(curTan.y) < 0.99 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0)
      dN.crossVectors(ref, curTan).normalize()
      dB.crossVectors(curTan, dN)

      // Point 0: hourAngle = 0 → cos=1, sin=0
      hourPos[0] = dayHR[0] + hOff * dN.x
      hourPos[1] = dayHR[1] + hOff * dN.y
      hourPos[2] = dayHR[2] + hOff * dN.z
      tmpColor.setHSL(0, 1, 0.5)
      hourCol[0] = tmpColor.r; hourCol[1] = tmpColor.g; hourCol[2] = tmpColor.b

      prevTan.copy(curTan)

      for (let i = 1; i < totalHourPts; i++) {
        const t = i / totalHourSegs

        // Tangent via central finite diff
        const pi = Math.max(0, i - 1) * 3
        const ni = Math.min(totalHourPts - 1, i + 1) * 3
        curTan.set(dayHR[ni] - dayHR[pi], dayHR[ni + 1] - dayHR[pi + 1], dayHR[ni + 2] - dayHR[pi + 2]).normalize()

        // Parallel transport: rotate normal to match new tangent
        crossV.crossVectors(prevTan, curTan)
        const cl = crossV.length()
        if (cl > 1e-10) {
          crossV.divideScalar(cl)
          const ang = Math.acos(THREE.MathUtils.clamp(prevTan.dot(curTan), -1, 1))
          dN.applyMatrix4(rotMat.makeRotationAxis(crossV, ang))
        }
        dB.crossVectors(curTan, dN)

        // Hour-coil offset
        const hA = t * totalHourTurns * Math.PI * 2
        const cH = Math.cos(hA), sH = Math.sin(hA)
        const j = i * 3
        hourPos[j]     = dayHR[j]     + hOff * (cH * dN.x + sH * dB.x)
        hourPos[j + 1] = dayHR[j + 1] + hOff * (cH * dN.y + sH * dB.y)
        hourPos[j + 2] = dayHR[j + 2] + hOff * (cH * dN.z + sH * dB.z)

        // Hue cycles once per day
        const dayFrac = (t * totalDayTurns) % 1
        tmpColor.setHSL(dayFrac, 1, 0.5)
        hourCol[j] = tmpColor.r; hourCol[j + 1] = tmpColor.g; hourCol[j + 2] = tmpColor.b

        prevTan.copy(curTan)
      }

      const hourGeom = new THREE.BufferGeometry()
      hourGeom.setAttribute('position', new THREE.BufferAttribute(hourPos, 3))
      hourGeom.setAttribute('color', new THREE.BufferAttribute(hourCol, 3))
      rt.hourCoilLine = new THREE.Line(hourGeom, rt.lineMaterial)
      rt.scene.add(rt.hourCoilLine)
    }

    const init = async () => {
      if (!navigator.gpu) {
        document.body.innerHTML =
          '<div style="display:flex;align-items:center;justify-content:center;height:100vh;color:#fff;background:#111;font-family:system-ui;font-size:1.5rem;">WebGPU is not supported in this browser.</div>'
        return
      }

      const width = window.innerWidth
      const height = window.innerHeight

      const renderer = new WebGPURenderer({ canvas, antialias: true })
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
      renderer.setSize(width, height, false)
      await renderer.init()

      const isWebGPU = (renderer as any).backend?.isWebGPUBackend
      if (!isWebGPU) {
        document.body.innerHTML =
          '<div style="display:flex;align-items:center;justify-content:center;height:100vh;color:#fff;background:#111;font-family:system-ui;font-size:1.5rem;">WebGPU backend failed to initialize.</div>'
        renderer.dispose()
        return
      }

      if (disposed) {
        renderer.dispose()
        return
      }

      const scene = new THREE.Scene()
      scene.background = new THREE.Color(0x111111)

      const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 100)
      camera.position.set(3, 3, 5)

      const controls = new OrbitControls(camera, renderer.domElement)
      controls.enableDamping = true
      controls.dampingFactor = 0.08

      const dirLight = new THREE.DirectionalLight(0xffffff, 2)
      dirLight.position.set(3, 5, 4)
      scene.add(dirLight)

      const ambLight = new THREE.AmbientLight(0xffffff, 0.3)
      scene.add(ambLight)

      const material = new MeshStandardNodeMaterial({ roughness: 0.4, metalness: 0.1 })
      material.colorNode = coilColorNode

      const lineMaterial = new THREE.LineBasicMaterial({ vertexColors: true })

      const pane = new Pane({ title: 'Coil' })

      const rt: Runtime = {
        renderer, scene, camera, controls,
        coilMesh: null, dayCoilLine: null, hourCoilLine: null,
        material, lineMaterial, pane,
      }
      runtimeRef.current = rt

      rebuild(rt)

      pane.addBinding(params, 'turns', { min: 1, max: 20, step: 1 })
        .on('change', () => rebuild(rt))
      pane.addBinding(params, 'tubeRadius', { min: 0.01, max: 0.5, step: 0.01, label: 'tube radius' })
        .on('change', () => rebuild(rt))
      pane.addBinding(params, 'coilRadius', { min: 0.1, max: 5, step: 0.1, label: 'coil radius' })
        .on('change', () => rebuild(rt))
      pane.addBinding(params, 'turnSpacing', { min: 0.1, max: 5, step: 0.1, label: 'turn spacing' })
        .on('change', () => rebuild(rt))
      pane.addBinding(params, 'outerOffset', { min: 0.05, max: 1, step: 0.01, label: 'day offset' })
        .on('change', () => rebuild(rt))
      pane.addBinding(params, 'hourOffset', { min: 0.001, max: 0.1, step: 0.001, label: 'hour offset' })
        .on('change', () => rebuild(rt))

      const animate = () => {
        if (disposed) return
        rafRef.current = requestAnimationFrame(animate)
        const r = runtimeRef.current!
        r.controls.update()
        r.renderer.render(r.scene, r.camera)
      }
      rafRef.current = requestAnimationFrame(animate)
    }

    init()

    const handleResize = () => {
      const rt = runtimeRef.current
      if (!rt) return
      const w = window.innerWidth
      const h = window.innerHeight
      rt.renderer.setSize(w, h, false)
      rt.camera.aspect = w / h
      rt.camera.updateProjectionMatrix()
    }
    window.addEventListener('resize', handleResize)

    return () => {
      disposed = true
      window.removeEventListener('resize', handleResize)
      cancelAnimationFrame(rafRef.current)
      const rt = runtimeRef.current
      if (rt) {
        rt.pane.dispose()
        rt.controls.dispose()
        if (rt.coilMesh) rt.coilMesh.geometry.dispose()
        if (rt.dayCoilLine) rt.dayCoilLine.geometry.dispose()
        if (rt.hourCoilLine) rt.hourCoilLine.geometry.dispose()
        rt.material.dispose()
        rt.lineMaterial.dispose()
        rt.renderer.dispose()
        runtimeRef.current = null
      }
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }}
    />
  )
}
