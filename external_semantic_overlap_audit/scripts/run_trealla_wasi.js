#!/usr/bin/env node

const fs = require("node:fs");
const { WASI } = require("node:wasi");

const [wasmPath, ...prologArgs] = process.argv.slice(2);
if (!wasmPath) {
  process.stderr.write("usage: run_trealla_wasi.js <tpl.wasm> [tpl args...]\n");
  process.exit(2);
}

const wasi = new WASI({
  version: "preview1",
  args: ["tpl", ...prologArgs],
  env: process.env,
  preopens: { ".": "." },
});

WebAssembly.instantiate(fs.readFileSync(wasmPath), wasi.getImportObject())
  .then(({ instance }) => wasi.start(instance))
  .catch((error) => {
    process.stderr.write(`${error.stack || error}\n`);
    process.exit(1);
  });
