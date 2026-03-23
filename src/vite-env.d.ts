/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_CLOCKSTORE_WS_URL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
