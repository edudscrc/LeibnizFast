/*
 * generator.cpp — CUDA 2D wave equation simulation -> ZeroMQ PUSH
 *
 * Simulates a 2D wave field on a CUDA-capable NVIDIA GPU and streams frames
 * over ZMQ PUSH. A Python bridge (bridge.py) relays frames to WebSocket clients.
 *
 * Two ZMQ sockets:
 *   tcp://127.0.0.1:5555  PUSH — binary frame data out to bridge
 *   tcp://127.0.0.1:5556  PULL — control messages in from bridge
 *
 * Control message: 4-byte little-endian uint32 = new grid size N.
 * The simulation reinitializes immediately with the new size.
 *
 * Protocol v1: plain float32 chunks (magic 0x4C465A01, 32-byte header):
 *
 *   Offset  0: magic        = 0x4C465A01
 *   Offset  4: total_rows   (N)
 *   Offset  8: cols         (N)
 *   Offset 12: frame_id
 *   Offset 16: chunk_index  (0-based)
 *   Offset 20: total_chunks
 *   Offset 24: row_start
 *   Offset 28: chunk_rows
 *   Offset 32: float32[chunk_rows x cols]  row-major grid data
 *
 * Build:
 *   apt install libzmq3-dev nvidia-cuda-toolkit
 *   nvcc -x cu -std=c++17 -O2 -o generator generator.cpp -lzmq
 *
 * Run:
 *   ./generator --check-cuda            # verify CUDA device/driver
 *   ./generator                         # 512x512, CUDA protocol source
 *   ./generator --size 4096             # larger grid
 *   ./generator --chunks 4              # 4 ZMQ messages per frame
 *   ./generator --debug                 # per-frame performance logging
 */

#include <cuda_runtime.h>
#include <zmq.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// ---- Constants -------------------------------------------------------

static constexpr uint32_t CHUNK_MAGIC = 0x4C465A01u;
static constexpr int CHUNK_HEADER_BYTES = 32;  // 8 x uint32

static constexpr float C_SPEED = 1.0f;
static constexpr float DAMPING = 0.999f;
static constexpr int IMPULSE_INTERVAL = 60;
static constexpr float IMPULSE_AMP = 0.5f;
static constexpr int N_MIN = 4;
static constexpr int N_MAX = 8192;
static constexpr int CHUNKS_MAX = 64;

// 16x16 keeps 256 CUDA threads per block, a good general-purpose occupancy
// target for this stencil while preserving simple 2D indexing.
static constexpr int CUDA_BLOCK_X = 16;
static constexpr int CUDA_BLOCK_Y = 16;

// ---- Signal handling -------------------------------------------------

static volatile bool g_running = true;
static void handle_sigint(int) { g_running = false; }

// ---- CUDA kernels ----------------------------------------------------

__device__ __forceinline__ int idx(int i, int j, int n) { return i * n + j; }

__global__ void wave_step_kernel(
    const float* u_prev,
    const float* u_curr,
    float* u_next,
    int n,
    float r) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= n || j >= n) {
    return;
  }

  const int center = idx(i, j, n);
  if (i == 0 || j == 0 || i == n - 1 || j == n - 1) {
    u_next[center] = 0.0f;
    return;
  }

  const float lap = u_curr[idx(i + 1, j, n)] + u_curr[idx(i - 1, j, n)] +
                    u_curr[idx(i, j + 1, n)] + u_curr[idx(i, j - 1, n)] -
                    4.0f * u_curr[center];
  u_next[center] =
      DAMPING * (2.0f * u_curr[center] - u_prev[center] + r * lap);
}

__global__ void impulse_kernel(float* u_next, int n, int i, int j, float amp) {
  if (i > 0 && j > 0 && i < n - 1 && j < n - 1) {
    u_next[idx(i, j, n)] += amp;
  }
}

// ---- CUDA helpers ----------------------------------------------------

std::string cuda_error(cudaError_t err) {
  return std::string(cudaGetErrorName(err)) + ": " + cudaGetErrorString(err);
}

bool cuda_check(cudaError_t err, const char* label, std::string* error) {
  if (err == cudaSuccess) {
    return true;
  }
  std::ostringstream oss;
  oss << label << " failed: " << cuda_error(err);
  *error = oss.str();
  return false;
}

bool check_cuda_available(std::string* error) {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess) {
    std::ostringstream oss;
    oss << "CUDA is required for the cpp-stream example, but CUDA device "
        << "enumeration failed: " << cuda_error(err)
        << ". Install a compatible NVIDIA driver and use a CUDA-capable GPU.";
    *error = oss.str();
    return false;
  }

  if (count <= 0) {
    *error =
        "CUDA is required for the cpp-stream example, but no CUDA-capable GPU "
        "was found. Use an NVIDIA GPU with a working CUDA driver.";
    return false;
  }

  cudaDeviceProp prop {};
  if (!cuda_check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties",
                  error)) {
    return false;
  }
  if (!cuda_check(cudaSetDevice(0), "cudaSetDevice", error)) {
    return false;
  }
  if (!cuda_check(cudaFree(nullptr), "CUDA context initialization", error)) {
    return false;
  }

  std::cout << "CUDA device: " << prop.name << " (compute capability "
            << prop.major << "." << prop.minor << ")\n";
  return true;
}

bool cuda_alloc_zero(float** ptr, std::size_t bytes, std::string* error) {
  *ptr = nullptr;
  if (!cuda_check(cudaMalloc(reinterpret_cast<void**>(ptr), bytes),
                  "cudaMalloc", error)) {
    return false;
  }
  if (!cuda_check(cudaMemset(*ptr, 0, bytes), "cudaMemset", error)) {
    cudaFree(*ptr);
    *ptr = nullptr;
    return false;
  }
  return true;
}

// ---- Simulation state ------------------------------------------------

struct SimState {
  int n = 0;
  int n_chunks = 0;
  float dx = 0.0f;
  float dt = 0.0f;
  float r = 0.0f;

  float* d_prev = nullptr;
  float* d_curr = nullptr;
  float* d_next = nullptr;

  // send_buf: CHUNK_HEADER_BYTES + max_chunk_float_bytes
  std::vector<uint8_t> send_buf;

  explicit SimState(int chunks) : n_chunks(chunks) {}

  ~SimState() { release(); }

  void release() {
    cudaFree(d_prev);
    cudaFree(d_curr);
    cudaFree(d_next);
    d_prev = nullptr;
    d_curr = nullptr;
    d_next = nullptr;
  }

  bool resize(int new_n, std::string* error) {
    const std::size_t cells = static_cast<std::size_t>(new_n) * new_n;
    const std::size_t bytes = cells * sizeof(float);

    float* new_prev = nullptr;
    float* new_curr = nullptr;
    float* new_next = nullptr;
    if (!cuda_alloc_zero(&new_prev, bytes, error) ||
        !cuda_alloc_zero(&new_curr, bytes, error) ||
        !cuda_alloc_zero(&new_next, bytes, error)) {
      cudaFree(new_prev);
      cudaFree(new_curr);
      cudaFree(new_next);
      return false;
    }

    release();
    d_prev = new_prev;
    d_curr = new_curr;
    d_next = new_next;
    n = new_n;
    dx = 1.0f / new_n;
    dt = 0.4f * dx / C_SPEED;
    r = (C_SPEED * dt / dx) * (C_SPEED * dt / dx);

    const int max_chunk_rows = (new_n + n_chunks - 1) / n_chunks;
    const std::size_t float_bytes =
        static_cast<std::size_t>(max_chunk_rows) * new_n * sizeof(float);
    send_buf.resize(CHUNK_HEADER_BYTES + float_bytes);
    return true;
  }

  bool step(uint32_t frame_id, double* sim_ms, std::string* error) {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    const dim3 block(CUDA_BLOCK_X, CUDA_BLOCK_Y);
    const dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    wave_step_kernel<<<grid, block>>>(d_prev, d_curr, d_next, n, r);
    if (!cuda_check(cudaGetLastError(), "wave_step_kernel launch", error)) {
      return false;
    }

    if (frame_id % IMPULSE_INTERVAL == 0) {
      const int pi = 1 + std::rand() % (n - 2);
      const int pj = 1 + std::rand() % (n - 2);
      impulse_kernel<<<1, 1>>>(d_next, n, pi, pj, IMPULSE_AMP);
      if (!cuda_check(cudaGetLastError(), "impulse_kernel launch", error)) {
        return false;
      }
    }

    if (!cuda_check(cudaDeviceSynchronize(), "CUDA simulation step", error)) {
      return false;
    }

    std::swap(d_prev, d_curr);
    std::swap(d_curr, d_next);

    using duration_ms = std::chrono::duration<double, std::milli>;
    *sim_ms = std::chrono::duration_cast<duration_ms>(clock::now() - t0).count();
    return true;
  }
};

// ---- Main ------------------------------------------------------------

int main(int argc, char* argv[]) {
  int initial_n = 512;
  int n_chunks = 4;
  bool debug = false;
  bool check_cuda = false;

  for (int a = 1; a < argc; ++a) {
    const std::string arg(argv[a]);
    if (arg == "--size" && a + 1 < argc) {
      initial_n = std::stoi(argv[++a]);
      if (initial_n < N_MIN || initial_n > N_MAX) {
        std::cerr << "size must be " << N_MIN << ".." << N_MAX << "\n";
        return 1;
      }
    } else if (arg == "--chunks" && a + 1 < argc) {
      n_chunks = std::stoi(argv[++a]);
      if (n_chunks < 1 || n_chunks > CHUNKS_MAX) {
        std::cerr << "chunks must be 1.." << CHUNKS_MAX << "\n";
        return 1;
      }
    } else if (arg == "--debug") {
      debug = true;
    } else if (arg == "--check-cuda") {
      check_cuda = true;
    }
  }

  std::string error;
  if (!check_cuda_available(&error)) {
    std::cerr << error << "\n";
    return 1;
  }
  if (check_cuda) {
    return 0;
  }

  signal(SIGINT, handle_sigint);
  std::srand(static_cast<unsigned>(std::time(nullptr)));

  // ---- ZMQ setup ---------------------------------------------------

  void* ctx = zmq_ctx_new();

  void* sock_data = zmq_socket(ctx, ZMQ_PUSH);
  {
    int sndhwm = 2 * n_chunks;
    zmq_setsockopt(sock_data, ZMQ_SNDHWM, &sndhwm, sizeof(sndhwm));
    int linger = 0;
    zmq_setsockopt(sock_data, ZMQ_LINGER, &linger, sizeof(linger));
    if (zmq_bind(sock_data, "tcp://127.0.0.1:5555") != 0) {
      std::cerr << "zmq_bind(data) failed: " << zmq_strerror(zmq_errno())
                << "\n";
      zmq_close(sock_data);
      zmq_ctx_destroy(ctx);
      return 1;
    }
  }

  void* sock_ctrl = zmq_socket(ctx, ZMQ_PULL);
  {
    int rcvhwm = 4;
    zmq_setsockopt(sock_ctrl, ZMQ_RCVHWM, &rcvhwm, sizeof(rcvhwm));
    int linger = 0;
    zmq_setsockopt(sock_ctrl, ZMQ_LINGER, &linger, sizeof(linger));
    if (zmq_bind(sock_ctrl, "tcp://127.0.0.1:5556") != 0) {
      std::cerr << "zmq_bind(ctrl) failed: " << zmq_strerror(zmq_errno())
                << "\n";
      zmq_close(sock_data);
      zmq_close(sock_ctrl);
      zmq_ctx_destroy(ctx);
      return 1;
    }
  }

  SimState sim(n_chunks);
  if (!sim.resize(initial_n, &error)) {
    std::cerr << "Failed to allocate CUDA simulation buffers: " << error << "\n";
    zmq_close(sock_data);
    zmq_close(sock_ctrl);
    zmq_ctx_destroy(ctx);
    return 1;
  }

  std::cout << "ZMQ PUSH (data) bound to tcp://127.0.0.1:5555\n";
  std::cout << "ZMQ PULL (ctrl) bound to tcp://127.0.0.1:5556\n";
  std::cout << "CUDA wave sim: " << initial_n << "x" << initial_n
            << "  n_chunks=" << n_chunks << "\n";
  if (debug) {
    std::cout << "Debug mode enabled — per-frame performance logging active\n";
  }
  std::cout << "Press Ctrl+C to stop.\n";

  uint32_t frame_id = 0;
  const auto frame_duration = std::chrono::milliseconds(33);  // ~30 FPS

  auto ms_since = [](std::chrono::steady_clock::time_point t) -> double {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now() - t).count();
  };

  while (g_running) {
    const auto frame_start = std::chrono::steady_clock::now();
    const int n = sim.n;

    // ---- Poll for resize control message (non-blocking) ----------
    {
      uint8_t ctrl_buf[4];
      if (zmq_recv(sock_ctrl, ctrl_buf, sizeof(ctrl_buf), ZMQ_DONTWAIT) == 4) {
        uint32_t new_n = 0;
        std::memcpy(&new_n, ctrl_buf, 4);
        if (new_n >= static_cast<uint32_t>(N_MIN) &&
            new_n <= static_cast<uint32_t>(N_MAX) &&
            new_n != static_cast<uint32_t>(sim.n)) {
          std::cout << "Resize: " << sim.n << " -> " << new_n << "\n";
          if (sim.resize(static_cast<int>(new_n), &error)) {
            frame_id = 0;
            continue;
          }
          std::cerr << "Resize failed; keeping current simulation: " << error
                    << "\n";
        }
      }
    }

    // ---- Wave equation step on CUDA --------------------------------
    double sim_ms = 0.0;
    if (!sim.step(frame_id, &sim_ms, &error)) {
      std::cerr << "CUDA simulation failed: " << error << "\n";
      break;
    }

    // ---- Send frame as n_chunks ZMQ messages ---------------------
    const auto t_send = std::chrono::steady_clock::now();
    const int rows_per_chunk = (n + sim.n_chunks - 1) / sim.n_chunks;

    for (int c = 0; c < sim.n_chunks; ++c) {
      const int row_start = c * rows_per_chunk;
      const int chunk_rows = std::min(rows_per_chunk, n - row_start);
      if (chunk_rows <= 0) {
        continue;
      }
      const int chunk_cells = chunk_rows * n;
      const std::size_t data_bytes =
          static_cast<std::size_t>(chunk_cells) * sizeof(float);

      auto* h = reinterpret_cast<uint32_t*>(sim.send_buf.data());
      h[0] = CHUNK_MAGIC;
      h[1] = static_cast<uint32_t>(n);
      h[2] = static_cast<uint32_t>(n);
      h[3] = frame_id;
      h[4] = static_cast<uint32_t>(c);
      h[5] = static_cast<uint32_t>(sim.n_chunks);
      h[6] = static_cast<uint32_t>(row_start);
      h[7] = static_cast<uint32_t>(chunk_rows);

      if (!cuda_check(
              cudaMemcpy(sim.send_buf.data() + CHUNK_HEADER_BYTES,
                         sim.d_curr + static_cast<std::size_t>(row_start) * n,
                         data_bytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy frame chunk", &error)) {
        std::cerr << "CUDA readback failed: " << error << "\n";
        g_running = false;
        break;
      }

      zmq_send(sock_data, sim.send_buf.data(),
               CHUNK_HEADER_BYTES + data_bytes, ZMQ_DONTWAIT);
    }
    const double send_ms = ms_since(t_send);

    if (debug) {
      const double total_ms = ms_since(frame_start);
      std::cout << "[perf] frame=" << frame_id << "  cuda_sim=" << sim_ms
                << "ms"
                << "  send=" << send_ms << "ms"
                << "  total=" << total_ms << "ms"
                << "  size=" << n << "x" << n << "  chunks=" << sim.n_chunks
                << "\n";
    }

    ++frame_id;

    // ---- Frame rate throttle -------------------------------------
    const auto elapsed = std::chrono::steady_clock::now() - frame_start;
    if (elapsed < frame_duration) {
      std::this_thread::sleep_for(frame_duration - elapsed);
    }
  }

  std::cout << "\nShutting down after " << frame_id << " frames.\n";
  zmq_close(sock_data);
  zmq_close(sock_ctrl);
  zmq_ctx_destroy(ctx);
  return 0;
}
