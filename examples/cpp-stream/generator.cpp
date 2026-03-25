/*
 * generator.cpp — 2D wave equation simulation -> ZeroMQ PUSH
 *
 * Simulates a 2D wave field and streams frames over ZMQ PUSH.
 * A Python bridge (bridge.py) relays frames to WebSocket clients.
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
 *   apt install libzmq3-dev             # Debian/Ubuntu
 *   brew install zeromq                 # macOS
 *   g++ -std=c++17 -O2 -o generator generator.cpp -lzmq
 *
 * Run:
 *   ./generator                          # 512x512, plain protocol
 *   ./generator --size 4096             # larger grid
 *   ./generator --chunks 4             # 4 ZMQ messages per frame
 *   ./generator --debug                # per-frame performance logging
 */

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

// ---- Signal handling -------------------------------------------------

static volatile bool g_running = true;
static void handle_sigint(int) { g_running = false; }

// ---- Grid index helper -----------------------------------------------

inline int idx(int i, int j, int N) { return i * N + j; }

// ---- Simulation state ------------------------------------------------

struct SimState {
  int N;
  int n_chunks;
  float dx, dt, r;

  std::vector<float> u_prev_buf;
  std::vector<float> u_curr_buf;
  std::vector<float> u_next_buf;

  float* u_prev() { return u_prev_buf.data(); }
  float* u_curr() { return u_curr_buf.data(); }
  float* u_next() { return u_next_buf.data(); }

  // send_buf: CHUNK_HEADER_BYTES + max_chunk_float_bytes
  std::vector<uint8_t> send_buf;

  explicit SimState(int n, int chunks) : n_chunks(chunks) { resize(n); }

  void resize(int new_N) {
    N = new_N;
    dx = 1.0f / new_N;
    dt = 0.4f * dx / C_SPEED;
    r = (C_SPEED * dt / dx) * (C_SPEED * dt / dx);

    const int cells = new_N * new_N;
    u_prev_buf.assign(cells, 0.0f);
    u_curr_buf.assign(cells, 0.0f);
    u_next_buf.assign(cells, 0.0f);

    const int max_chunk_rows = (new_N + n_chunks - 1) / n_chunks;
    const std::size_t float_bytes =
        static_cast<std::size_t>(max_chunk_rows) * new_N * sizeof(float);

    send_buf.resize(CHUNK_HEADER_BYTES + float_bytes);
  }
};

// ---- Main ------------------------------------------------------------

int main(int argc, char* argv[]) {
  int initial_N = 512;
  int n_chunks = 4;
  bool g_debug = false;

  for (int a = 1; a < argc; ++a) {
    const std::string arg(argv[a]);
    if (arg == "--size" && a + 1 < argc) {
      initial_N = std::stoi(argv[++a]);
      if (initial_N < N_MIN || initial_N > N_MAX) {
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
      g_debug = true;
    }
  }

  signal(SIGINT, handle_sigint);
  srand(static_cast<unsigned>(time(nullptr)));

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

  std::cout << "ZMQ PUSH (data) bound to tcp://127.0.0.1:5555\n";
  std::cout << "ZMQ PULL (ctrl) bound to tcp://127.0.0.1:5556\n";
  std::cout << "Wave sim: " << initial_N << "x" << initial_N
            << "  n_chunks=" << n_chunks << "\n";
  if (g_debug)
    std::cout << "Debug mode enabled — per-frame performance logging active\n";
  std::cout << "Press Ctrl+C to stop.\n";

  SimState sim(initial_N, n_chunks);
  uint32_t frame_id = 0;
  const auto frame_duration = std::chrono::milliseconds(33);  // ~30 FPS

  auto ms_since = [](std::chrono::steady_clock::time_point t) -> double {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now() - t).count();
  };

  while (g_running) {
    const auto frame_start = std::chrono::steady_clock::now();
    const int N = sim.N;

    // ---- Poll for resize control message (non-blocking) ----------
    {
      uint8_t ctrl_buf[4];
      if (zmq_recv(sock_ctrl, ctrl_buf, sizeof(ctrl_buf), ZMQ_DONTWAIT) == 4) {
        uint32_t new_N = 0;
        std::memcpy(&new_N, ctrl_buf, 4);
        if (new_N >= static_cast<uint32_t>(N_MIN) &&
            new_N <= static_cast<uint32_t>(N_MAX)) {
          std::cout << "Resize: " << N << " -> " << new_N << "\n";
          sim.resize(static_cast<int>(new_N));
          frame_id = 0;
          continue;
        }
      }
    }

    // ---- Wave equation step (interior points only) ---------------
    const auto t_sim = std::chrono::steady_clock::now();
    float* uc = sim.u_curr();
    float* up = sim.u_prev();
    float* un = sim.u_next();
    for (int i = 1; i < N - 1; ++i) {
      for (int j = 1; j < N - 1; ++j) {
        const float lap = uc[idx(i + 1, j, N)] + uc[idx(i - 1, j, N)] +
                          uc[idx(i, j + 1, N)] + uc[idx(i, j - 1, N)] -
                          4.0f * uc[idx(i, j, N)];
        un[idx(i, j, N)] = DAMPING * (2.0f * uc[idx(i, j, N)] -
                                      up[idx(i, j, N)] + sim.r * lap);
      }
    }
    const double sim_ms = ms_since(t_sim);

    // ---- Periodic impulse ----------------------------------------
    if (frame_id % IMPULSE_INTERVAL == 0) {
      const int pi = 1 + rand() % (N - 2);
      const int pj = 1 + rand() % (N - 2);
      un[idx(pi, pj, N)] += IMPULSE_AMP;
    }

    // ---- Rotate time levels (O(1) swap) --------------------------
    std::swap(sim.u_prev_buf, sim.u_curr_buf);
    std::swap(sim.u_curr_buf, sim.u_next_buf);

    // ---- Send frame as n_chunks ZMQ messages ---------------------
    const auto t_send = std::chrono::steady_clock::now();
    const int rows_per_chunk = (N + sim.n_chunks - 1) / sim.n_chunks;

    for (int c = 0; c < sim.n_chunks; ++c) {
      const int row_start = c * rows_per_chunk;
      const int chunk_rows = std::min(rows_per_chunk, N - row_start);
      const int chunk_cells = chunk_rows * N;
      const std::size_t data_bytes =
          static_cast<std::size_t>(chunk_cells) * sizeof(float);

      auto* h = reinterpret_cast<uint32_t*>(sim.send_buf.data());
      h[0] = CHUNK_MAGIC;
      h[1] = static_cast<uint32_t>(N);
      h[2] = static_cast<uint32_t>(N);
      h[3] = frame_id;
      h[4] = static_cast<uint32_t>(c);
      h[5] = static_cast<uint32_t>(sim.n_chunks);
      h[6] = static_cast<uint32_t>(row_start);
      h[7] = static_cast<uint32_t>(chunk_rows);

      std::memcpy(sim.send_buf.data() + CHUNK_HEADER_BYTES,
                  sim.u_curr() + row_start * N, data_bytes);

      zmq_send(sock_data, sim.send_buf.data(),
               CHUNK_HEADER_BYTES + data_bytes, ZMQ_DONTWAIT);
    }
    const double send_ms = ms_since(t_send);

    if (g_debug) {
      const double total_ms = ms_since(frame_start);
      std::cout << "[perf] frame=" << frame_id << "  sim=" << sim_ms << "ms"
                << "  send=" << send_ms << "ms"
                << "  total=" << total_ms << "ms"
                << "  size=" << N << "x" << N << "  chunks=" << sim.n_chunks
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
