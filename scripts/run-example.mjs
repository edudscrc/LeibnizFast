#!/usr/bin/env node

import { createReadStream, existsSync, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, join, normalize, resolve } from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = resolve(fileURLToPath(new URL('..', import.meta.url)));
const example = process.argv[2] ?? 'chart';
const debug = process.argv.includes('--debug');

const examples = new Map([
  ['chart', { path: '/examples/chart/' }],
  [
    'cpp-stream',
    {
      path: '/examples/cpp-stream/',
      generatorDir: 'examples/cpp-stream',
      bridge: 'examples/cpp-stream/bridge.py',
      compiler: 'nvcc',
      requiresCuda: true,
    },
  ],
  [
    'waterfall',
    {
      path: '/examples/waterfall/',
      generatorDir: 'examples/waterfall',
      bridge: 'examples/waterfall/bridge.py',
      compiler: 'g++',
    },
  ],
  ['line', { path: '/examples/line/' }]
]);

const config = examples.get(example);
if (!config) {
  console.error(
    `Unknown example "${example}". Choose one of: ${[...examples.keys()].join(', ')}`,
  );
  process.exit(1);
}

const children = new Set();
let server = null;
let shuttingDown = false;

function run(command, args, options = {}) {
  return new Promise((resolveRun, reject) => {
    console.log(`$ ${[command, ...args].join(' ')}`);
    const child = spawn(command, args, {
      cwd: root,
      stdio: 'inherit',
      ...options,
    });
    child.on('error', reject);
    child.on('exit', (code, signal) => {
      if (code === 0) {
        resolveRun();
      } else {
        reject(
          new Error(
            `${command} exited with ${code ?? `signal ${signal ?? 'unknown'}`}`,
          ),
        );
      }
    });
  });
}

function start(command, args, options = {}) {
  console.log(`$ ${[command, ...args].join(' ')}`);
  const child = spawn(command, args, {
    cwd: root,
    stdio: 'inherit',
    ...options,
  });
  children.add(child);
  child.on('exit', () => children.delete(child));
  return child;
}

async function ensureNodeDeps() {
  if (existsSync(join(root, 'node_modules'))) return;
  console.log('node_modules not found; running npm install...');
  await run('npm', ['install']);
}

async function ensureBuild() {
  await run('npm', ['run', 'build']);
}

async function ensurePythonEnv() {
  const python = join(root, 'venv/bin/python');
  if (!existsSync(python)) {
    await run('python3', ['-m', 'venv', 'venv']);
  }
  await run(python, [
    '-m',
    'pip',
    'install',
    '-r',
    'examples/requirements.txt',
  ]);
  return python;
}

async function compileGenerator(generatorConfig) {
  const out = join(generatorConfig.generatorDir, 'generator');
  const src = join(generatorConfig.generatorDir, 'generator.cpp');
  if (generatorConfig.compiler === 'nvcc') {
    try {
      await run('nvcc', [
        '-x',
        'cu',
        '-std=c++17',
        '-O2',
        '-o',
        out,
        src,
        '-lzmq',
      ]);
    } catch (error) {
      throw new Error(
        `Failed to build the CUDA cpp-stream generator with nvcc. ` +
          `Install the NVIDIA CUDA Toolkit, libzmq headers, and ensure nvcc is on PATH.\n` +
          `${error instanceof Error ? error.message : error}`,
      );
    }
    return;
  }

  await run('g++', ['-std=c++17', '-O2', '-o', out, src, '-lzmq']);
}

async function preflightGenerator(generatorConfig) {
  if (!generatorConfig.requiresCuda) return;
  const generatorPath = join(root, generatorConfig.generatorDir, 'generator');
  try {
    await run(generatorPath, ['--check-cuda']);
  } catch (error) {
    throw new Error(
      `The cpp-stream example requires an NVIDIA CUDA-capable GPU and a working CUDA driver.\n` +
        `${error instanceof Error ? error.message : error}`,
    );
  }
}

function contentType(pathname) {
  switch (extname(pathname)) {
    case '.html':
      return 'text/html; charset=utf-8';
    case '.js':
      return 'text/javascript; charset=utf-8';
    case '.mjs':
      return 'text/javascript; charset=utf-8';
    case '.css':
      return 'text/css; charset=utf-8';
    case '.wasm':
      return 'application/wasm';
    case '.json':
      return 'application/json; charset=utf-8';
    case '.svg':
      return 'image/svg+xml';
    default:
      return 'application/octet-stream';
  }
}

function resolveRequest(urlPath) {
  const [rawPath = '/', rawQuery = ''] = (urlPath ?? '/').split('?');
  const decoded = decodeURIComponent(rawPath);
  const normalized = normalize(decoded).replace(/^(\.\.[/\\])+/, '');
  let filePath = join(root, normalized);
  if (!filePath.startsWith(root)) {
    return null;
  }
  if (existsSync(filePath) && statSync(filePath).isDirectory()) {
    if (!rawPath.endsWith('/')) {
      return {
        redirectPath: `${rawPath}/${rawQuery ? `?${rawQuery}` : ''}`,
      };
    }
    filePath = join(filePath, 'index.html');
  }
  return { filePath };
}

function startServer(preferredPort = 8080) {
  return new Promise((resolveServer, reject) => {
    const tryPort = (port) => {
      const candidate = createServer((req, res) => {
        const resolved = resolveRequest(req.url ?? '/');
        if (resolved?.redirectPath) {
          res.writeHead(308, { location: resolved.redirectPath });
          res.end();
          return;
        }

        const filePath = resolved?.filePath;
        if (
          !filePath ||
          !existsSync(filePath) ||
          statSync(filePath).isDirectory()
        ) {
          res.writeHead(404, { 'content-type': 'text/plain; charset=utf-8' });
          res.end('Not found');
          return;
        }

        res.writeHead(200, {
          'content-type': contentType(filePath),
          'cache-control': 'no-store',
        });
        createReadStream(filePath).pipe(res);
      });

      candidate.on('error', (error) => {
        if (error.code === 'EADDRINUSE') {
          tryPort(port + 1);
        } else {
          reject(error);
        }
      });
      candidate.listen(port, '127.0.0.1', () => {
        server = candidate;
        resolveServer(port);
      });
    };
    tryPort(preferredPort);
  });
}

async function startStreamingProcesses() {
  if (!config.generatorDir || !config.bridge) return;

  await compileGenerator(config);
  await preflightGenerator(config);
  const python = await ensurePythonEnv();

  const generatorArgs = debug ? ['--debug'] : [];
  const bridgeArgs = [config.bridge];
  if (debug) bridgeArgs.push('--debug');

  start(python, bridgeArgs);
  start(join(root, config.generatorDir, 'generator'), generatorArgs);
}

function cleanup() {
  if (shuttingDown) return;
  shuttingDown = true;

  for (const child of children) {
    child.kill('SIGTERM');
  }
  setTimeout(() => {
    for (const child of children) {
      child.kill('SIGKILL');
    }
    server?.close();
    process.exit(0);
  }, 1500).unref();

  if (children.size === 0) {
    server?.close();
    process.exit(0);
  }
}

process.on('SIGINT', cleanup);
process.on('SIGTERM', cleanup);

try {
  await ensureNodeDeps();
  await ensureBuild();
  await startStreamingProcesses();
  const port = await startServer();
  console.log('');
  console.log(`LeibnizFast ${example} example is running:`);
  console.log(`  http://localhost:${port}${config.path}`);
  console.log('');
  console.log('Press Ctrl+C to stop.');
} catch (error) {
  console.error(error instanceof Error ? error.message : error);
  for (const child of children) {
    child.kill('SIGTERM');
  }
  server?.close();
  process.exit(1);
}
